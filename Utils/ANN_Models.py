import torch
from torch import nn
import math
from Utils.PDEs_ScikitFEM import *
from Utils.Initializers import *


class Supervised_ANN(nn.Module):
    """
    Approximates deterministic target function f by neural network.
    neurons: List/Tuple specifying the layer dimensions.
    """

    def __init__(self, neurons, f, space_bounds, dev, activation=nn.ReLU(), lr=0.001, sigma=0.):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dims = neurons
        self.depth = len(neurons) - 1
        for i in range(self.depth - 1):
            self.layers.append(nn.Linear(neurons[i], neurons[i + 1]))
            self.layers.append(activation)
        self.layers.append(nn.Linear(neurons[-2], neurons[-1]))

        self.initializer = UniformValueSampler(neurons[0], space_bounds, dev)
        self.target_fn = f
        self.sigma = sigma  # variance of random measurement noise

        self.lr = lr

    def forward(self, x):
        for fc in self.layers:
            x = fc(x)
        return x

    def loss(self, x):
        u = self.target_fn(x)
        y = u + math.sqrt(self.sigma) * torch.randn_like(u)  # output contains random noise
        output = self.forward(x)
        l = (y - output).square().mean()
        return l

    def test_loss(self, x):
        y = self.forward(x)
        l = (y - self.target_fn(x)).square().mean().sqrt()
        return l


class Heat_PDE_ANN(nn.Module):
    """
    Implements deep Kolmogorov method
    (method described in "Solving the Kolmogorov PDE by Means of Deep Learning" by
    Christian Beck, Sebastian Becker, Philipp Grohs, Nor Jaafari, and Arnulf Jentzen)
    for standard heat PDE.
    neurons: List/Tuple specifying the layer dimensions of the considered ANN.
    phi: Function representing the initial value.
    """

    def __init__(self, neurons, phi, space_bounds, T, rho, dev, activation=nn.ReLU(), lr=0.0001, final_u=None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dims = neurons
        self.depth = len(neurons) - 1
        for i in range(self.depth - 1):
            self.layers.append(nn.Linear(neurons[i], neurons[i + 1]))
            self.layers.append(activation)
        self.layers.append(nn.Linear(neurons[-2], neurons[-1]))

        self.initializer = UniformValueSampler(neurons[0], space_bounds, dev)
        self.phi = phi
        self.rho = rho
        self.T = T
        self.space_bounds = space_bounds
        self.final_u = final_u

        self.lr = lr

    def forward(self, x):
        for fc in self.layers:
            x = fc(x)
        return x

    def loss(self, data):
        W = torch.randn_like(data)
        return (self.phi(math.sqrt(2 * self.rho * self.T) * W + data) - self.forward(data)).square().mean()

    def test_loss(self, data):
        output = self.forward(data)
        u_T = self.final_u(data)
        return ((u_T - output).square().mean() / u_T.square().mean()).sqrt()


class BlackScholes_ANN(nn.Module):
    """
    Implements deep Kolmogorov method for Black-Scholes PDE with correlated noise (sections 4.3 and 4.4 in Beck et al.).
    """

    def __init__(self, neurons, phi, space_bounds, T, c, r, Q, sigma, dev, activation=nn.ReLU(),
                 lr=0.0001, final_u=None, mc_samples=1024, test_size=10000, mc_rounds=100):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dims = neurons
        self.depth = len(neurons) - 1
        self.layers.append(nn.BatchNorm1d(neurons[0]))
        for i in range(self.depth - 1):
            self.layers.append(nn.Linear(neurons[i], neurons[i + 1]))
            self.layers.append(activation)
        self.layers.append(nn.Linear(neurons[-2], neurons[-1]))

        self.initializer = UniformValueSampler(neurons[0], space_bounds, dev)
        self.phi = phi
        self.x_test = self.initializer.sample(test_size)

        self.mu = r - c
        self.r = r
        self.sigma = sigma.to(dev)
        self.mc_samples = mc_samples
        self.T = T
        self.space_bounds = space_bounds
        self.final_u = final_u

        self.lr = lr

        L = np.linalg.cholesky(Q).transpose()
        self.sigma_norms = torch.Tensor(np.linalg.norm(L, axis=0)).to(dev)
        self.sigma_matrix = torch.Tensor(L).to(dev)  # sigma x sigma* = Q (Cholesky decomposition)

        if self.final_u is None:  # approximate true solution by MC
            u_ref = torch.zeros([test_size, neurons[-1]]).to(dev)
            for i in range(mc_rounds):
                x = torch.stack([self.x_test for _ in range(self.mc_samples)])
                w = torch.matmul(torch.randn_like(x), self.sigma_matrix)
                u = self.phi(self.x_test * torch.exp((self.mu - 0.5 * (self.sigma * self.sigma_norms) ** 2) * self.T
                                                     + self.sigma * math.sqrt(self.T) * w))
                u = torch.mean(u, dim=0)
                u_ref += u
            self.u_test = u_ref / mc_rounds
        else:
            self.u_test = self.final_u(self.x_test)

    def forward(self, x):
        for fc in self.layers:
            x = fc(x)
        return x

    def loss(self, data):
        W = torch.randn_like(data)
        X = data * torch.exp((self.mu - 0.5 * (self.sigma * self.sigma_norms) ** 2) * self.T
                             + self.sigma * math.sqrt(self.T) * torch.matmul(W, self.sigma_matrix))
        return (self.phi(X) - self.forward(data)).square().mean()

    def test_loss(self, data):
        output = self.forward(self.x_test)

        return ((self.u_test - output).square().mean() / self.u_test.square().mean()).sqrt()


def HJB_compute_ref(x_test, mc_samples, mc_rounds, phi, T, sigma=math.sqrt(2)):
    # compute solution of HJB PDE via Monte-Carlo method and Cole-Hopf transform
    u_ref = torch.zeros([x_test.shape[0]]).to(x_test.device)
    for i in range(mc_rounds):
        x = torch.stack([x_test for _ in range(mc_samples)])
        w = torch.randn_like(x)
        u = torch.exp(- phi(x_test + sigma * math.sqrt(T) * w))
        u = torch.mean(u, dim=0)
        u_ref += u
    u_test = - torch.log(u_ref / mc_rounds)
    return u_test


class BSDE_Net(nn.Module):
    """Implements deep BSDE method described in E et al, arXiv:1706.04702."""
    def __init__(self, neurons, input_neurons, f, g, T, nt, dev, activation=nn.ReLU(), lr=0.0001,
                 batch_norm=True, test_size=1024, mc_samples=10000, mc_rounds=100, space_bounds=None, initial='N'):
        super().__init__()

        self.layers = nn.ModuleList([nn.ModuleList() for _ in range(nt)])
        self.network = nn.ModuleList()
        self.dims = neurons
        self.d_i = neurons[0]
        self.depth = len(neurons) - 1
        for i in range(len(input_neurons) - 2):
            self.network.append(nn.Linear(input_neurons[i], input_neurons[i + 1]))
            self.network.append(activation)
            if batch_norm:
                self.network.append(nn.BatchNorm1d(input_neurons[i + 1]))
        self.network.append(nn.Linear(input_neurons[-2], input_neurons[-1]))
        for k in range(nt):
            for i in range(self.depth - 1):
                self.layers[k].append(nn.Linear(neurons[i], neurons[i + 1]))
                self.layers[k].append(activation)
                if batch_norm:
                    self.layers[k].append(nn.BatchNorm1d(neurons[i + 1]))
            self.layers[k].append(nn.Linear(neurons[-2], neurons[-1]))

        if initial == 'N':
            self.initializer = NormalValueSampler(self.d_i, dev)
        elif initial == 'U' and space_bounds is not None:
            self.initializer = UniformValueSampler(self.d_i, space_bounds, dev)
        else:
            raise ValueError('Initializer must be N or U, and in the latter case space-bounds cannot be None.')
        self.phi = g  # initial condition
        self.f = f  # nonlinearity

        self.T = T
        self.nt = nt
        self.dt = T / nt

        self.lr = lr

        self.x_test = self.initializer.sample(test_size)
        self.u_test = HJB_compute_ref(self.x_test, mc_samples, mc_rounds, self.phi, T)
        # computing reference solution by Monte Carlo

    def forward(self, x):
        for fc in self.network:
            x = fc(x)
        return x

    def loss(self, data):
        Y0 = self.forward(data).squeeze()
        V0 = data
        for fc in self.layers[0]:
            V0 = fc(V0)
        w = torch.randn_like(data) * math.sqrt(2. * self.dt)
        Y = Y0 - self.f(Y0, V0) * self.dt + torch.sum(V0 * w, dim=-1)
        X = data + w
        for i in range(1, self.nt):
            V = X
            w = torch.randn_like(data) * math.sqrt(2. * self.dt)
            for fc in self.layers[i]:
                V = fc(V)

            Y = Y - self.f(Y, V) * self.dt + torch.sum(V * w, dim=-1)
            X = X + w

        return (Y.squeeze() - self.phi(X)).square().mean()

    def test_loss(self, x):
        output = self.forward(self.x_test).squeeze()

        return ((self.u_test - output).square().mean() / self.u_test.square().mean()).sqrt()


def solve_ricatti(A, B, Q, R, H, Sigma, T_end, nr_timesteps):
    """ computes approximate solution of Ricatti ODE
    P' = P B R^{-1} B^T P - A^T P - P A - Q
    P(T_end) = H
    using linear-implicit Euler """
    h = T_end / nr_timesteps
    d = H.shape[0]
    P = H
    inv_R = np.linalg.inv(R)
    inv_step = np.linalg.inv(
        h * (np.kron(np.eye(d), A.transpose()) + np.kron(A.transpose(), np.eye(d))) - np.eye(d * d))
    q = 0.
    for i in range(nr_timesteps):
        rhs = h * P @ B @ inv_R @ B.transpose() @ P - P - h * Q
        P_new = np.matmul(inv_step, rhs.flatten())
        P = np.reshape(P_new, (d, d))
        q += h * np.trace(Sigma.transpose() @ P @ Sigma)
    return P, q


class ControlNet(nn.Module):
    """Solving optimal control problem by approximating policy function in each time step by ANN."""

    def __init__(self, neurons, L, phi, mu, T, nt, dev, sigma=1., activation=nn.ReLU(), batch_norm=True,
                 space_bounds=None, initial='N', test_size=1024, mc_test=1024):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleList() for _ in range(nt - 1)])
        self.network = nn.ModuleList()
        self.dims = neurons
        self.d_i = neurons[0]
        self.depth = len(neurons) - 1

        for k in range(nt - 1):
            for i in range(self.depth - 1):
                self.layers[k].append(nn.Linear(neurons[i], neurons[i + 1]))
                self.layers[k].append(activation)
                if batch_norm:
                    self.layers[k].append(nn.BatchNorm1d(neurons[i + 1]))
            self.layers[k].append(nn.Linear(neurons[-2], neurons[-1]))
        self.dev = dev

        if initial == 'N':
            self.initializer = NormalValueSampler(self.d_i, dev)
        elif initial == 'U' and space_bounds is not None:
            self.initializer = UniformValueSampler(self.d_i, space_bounds, dev)
        else:
            raise ValueError('Initializer must be N or U, and in the latter case space-bounds cannot be None.')

        self.phi = phi  # initial condition/terminal cost
        self.f = L  # nonlinearity/running cost
        self.mu = mu  # drift term
        self.sigma = sigma  # diffusion coefficient (scalar)
        self.V0 = nn.Parameter(2. * torch.rand([self.d_i]) - 1.)

        self.T = T
        self.nt = nt
        self.dt = T / nt
        self.mc_test = mc_test

        self.x_test = self.initializer.sample(test_size)
        self.u_ref = None

    def loss(self, data):
        w = math.sqrt(self.dt) * torch.randn_like(data)
        l = self.f(data, self.V0) * self.dt * torch.ones([w.shape[0]]).to(self.dev)
        X = data + self.mu(data, self.V0) * self.dt + self.sigma * w
        for i in range(self.nt - 1):
            w = math.sqrt(self.dt) * torch.randn_like(data)
            V = X
            for fc in self.layers[i]:
                V = fc(V)
            l += self.f(X, V) * self.dt
            X = X + self.mu(X, V) * self.dt + self.sigma * w

        l += self.phi(X)
        return l.mean()

    def test_loss(self, x):
        result = torch.zeros_like(self.u_ref)
        for _ in range(self.mc_test):  # compute expected cost as average over SDE sample paths
            w = math.sqrt(self.dt) * torch.randn_like(self.x_test)
            l = self.f(self.x_test, self.V0) * self.dt * torch.ones([w.shape[0]]).to(self.dev)
            X = self.x_test + self.mu(self.x_test, self.V0) * self.dt + self.sigma * w
            for i in range(self.nt - 1):
                w = math.sqrt(self.dt) * torch.randn_like(self.x_test)
                V = X
                for fc in self.layers[i]:
                    V = fc(V)
                l += self.f(X, V) * self.dt
                X = X + self.mu(X, V) * self.dt + self.sigma * w

            l += self.phi(X)
            result += l
        result /= self.mc_test
        return (self.u_ref - result).square().mean().sqrt()


class PolyReg1d(nn.Module):
    """Modeling polynomial regression in one dimension, which leads to a convex problem (random feature model)."""

    def __init__(self, deg, P, n_i, dev, sigma=1., space_bounds=(-1., 1.)):
        super().__init__()

        self.deg = deg
        self.space_bounds = space_bounds
        self.P = P
        self.x_i = initial_values_sampler_uniform(n_i, 1, space_bounds)

        self.A = torch.cat([self.x_i ** k for k in range(self.deg + 1)], 1)

        self.params = torch.nn.Parameter(torch.randn(self.deg + 1))
        self.sigma = sigma
        self.initializer = UniformValueSampler(1, space_bounds, dev)

    def loss(self, data):
        y = self.P(data) + math.sqrt(self.sigma) * torch.randn_like(data)
        A = torch.cat([data ** k for k in range(self.deg + 1)], 1)  # design matrix (monomials)
        d = torch.matmul(A, self.params) - torch.squeeze(y)
        return d.square().mean()

    def test_loss(self, data):
        y = self.P(self.x_i)
        d = torch.matmul(self.A, self.params) - torch.squeeze(y)
        return d.square().mean().sqrt()


class SemilinHeat_PINN_2d(nn.Module):
    """
     Neural network to solve semilinear heat equation du/dt = alpha * Laplace(u) + nonlin(u) on rectangle [0, a] x [0, b]
     with either Dirichlet or periodic boundary conditions using PINN method.
    """

    def __init__(self, neurons, f, nonlin, alpha, space_bounds, T=1., boundary='D', test_discr=100, test_timesteps=500,
                 activation=nn.Tanh(), nonlin_name=None, torch_nonlin=None, train_points=10000, test_points=10000):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dims = neurons
        assert neurons[0] == 3
        self.depth = len(neurons) - 1
        for i in range(self.depth - 1):
            self.layers.append(nn.Linear(neurons[i], neurons[i + 1]))
            self.layers.append(activation)
        self.layers.append(nn.Linear(neurons[-2], neurons[-1]))
        self.f = f
        self.T = T
        self.alpha = alpha
        self.space_bounds = space_bounds
        self.spacetime_bounds = space_bounds + [T]
        self.boundary = boundary

        self.nonlin = nonlin
        self.nonlin_name = nonlin_name
        if torch_nonlin is None:
            self.torch_nonlin = nonlin
        else:
            self.torch_nonlin = torch_nonlin

        self.base = FEM_Basis(space_bounds, test_discr)
        self.ref_method = ReferenceMethod(Second_order_linear_implicit_RK_FEM, self.alpha, self.nonlin,
                                          self.nonlin_name,  (0.5, 0.5), 'LIRK2')
        self.pde = self.ref_method.create_ode(self.base)
        self.init_values = self.base.project_cont_function(f)
        self.final_sol = self.ref_method.compute_sol(T, self.init_values, test_timesteps, self.pde, self.base)
        self.u_t_fem = self.base.basis.interpolator(self.final_sol)

        initializer = RectangleValueSampler(3, self.spacetime_bounds)
        self.train_data = initializer.sample(train_points)
        self.initializer = DrawBatch(self.train_data)
        self.test_data = initializer.sample(test_points)
        self.test_sampler = DrawBatch(self.test_data)

    def forward(self, x):
        for fc in self.layers:
            x = fc(x)
        return x

    def loss(self, data):
        x = torch.Tensor(data[:, 0:2])
        t = torch.Tensor(data[:, 2:3])

        f0 = torch.Tensor(np.transpose(self.f(np.transpose(data[:, 0:2]))))
        x.requires_grad_()
        t.requires_grad_()

        x1 = x[:, 0:1]
        x2 = x[:, 1:2]

        x0 = torch.cat((x, torch.zeros_like(t)), 1)
        u0 = torch.squeeze(self.forward(x0))

        initial_loss = (u0 - f0).square().mean()

        u = self.forward(torch.cat((x1, x2, t), 1))

        u_x1 = torch.autograd.grad(u, x1, torch.ones_like(u), create_graph=True)[0]
        u_xx1 = torch.autograd.grad(u_x1, x1, torch.ones_like(u_x1), create_graph=True)[0]

        u_x2 = torch.autograd.grad(u, x2, torch.ones_like(u), create_graph=True)[0]
        u_xx2 = torch.autograd.grad(u_x2, x2, torch.ones_like(u_x2), create_graph=True)[0]

        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]

        loss = (self.alpha * (u_xx1 + u_xx2) + self.torch_nonlin(u) - u_t).square().mean()

        xa0 = torch.cat((torch.zeros_like(x1), x2, t), 1)
        xa1 = torch.cat((self.space_bounds[0] * torch.ones_like(x1), x2, t), 1)

        xb0 = torch.cat((x1, torch.zeros_like(x2), t), 1)
        xb1 = torch.cat((x1, self.space_bounds[1] * torch.ones_like(x2), t), 1)

        if self.boundary == 'D':
            boundary_loss = (self.forward(xa0)).square().mean() + (self.forward(xa1)).square().mean() \
                            + (self.forward(xb0)).square().mean() + (self.forward(xb1)).square().mean()
        elif self.boundary == 'P':
            boundary_loss = (self.forward(xa0) - self.forward(xa1)).square().mean() \
                            + (self.forward(xb0) - self.forward(xb1)).square().mean()
        else:
            raise ValueError('Boundary condition must be either "D" (Dirichlet) or "P" (Periodic)')

        loss_value = loss + boundary_loss + 2. * initial_loss
        return loss_value

    def test_loss(self, x):
        x = x[:, 0:2]
        y_t_fem = torch.Tensor(self.u_t_fem(np.transpose(x)))
        x = torch.Tensor(x)
        x_t = torch.cat((x, self.T * torch.ones_like(x[:, 0:1])), 1)
        y_t_net = torch.squeeze(self.forward(x_t))
        l2_err = (y_t_fem - y_t_net).square().mean()
        ref = y_t_fem.square().mean()

        return (l2_err / ref).sqrt()


class Burgers_PINN_1d(nn.Module):
    """ Neural network to solve equation du/dt = alpha * Laplace(u) - u * du / dx
    on interval (0, a) with 0 boundary conditions
    """
    def __init__(self, neurons, f, alpha, space_bounds, final_sol, T, dev, activation=nn.Tanh(),
                 train_points=1, test_points=1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dims = neurons
        assert neurons[0] == 2
        self.depth = len(neurons) - 1
        for i in range(self.depth - 1):
            self.layers.append(nn.Linear(neurons[i], neurons[i + 1]))
            self.layers.append(activation)
        self.layers.append(nn.Linear(neurons[-2], neurons[-1]))
        self.f = f
        self.alpha = alpha
        self.T = T
        self.space_bounds = space_bounds
        self.spacetime_bounds = space_bounds + [T]
        self.final_sol = final_sol
        self.dev = dev

        initializer = RectangleValueSampler(2, self.spacetime_bounds)

        self.train_data = initializer.sample(train_points)  # generate points for training and testing
        self.initializer = DrawBatch(self.train_data)
        self.test_data = initializer.sample(test_points)
        self.test_sampler = DrawBatch(self.test_data)

    def forward(self, x):
        for fc in self.layers:
            x = fc(x)
        return x

    def loss(self, data):
        x = torch.Tensor(data[:, 0:1]).to(self.dev)
        t = torch.Tensor(data[:, 1:2]).to(self.dev)
        x.requires_grad_()
        t.requires_grad_()

        u = self.forward(torch.cat((x, t), 1))
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]

        loss = (self.alpha * u_xx - u * u_x - u_t).square().mean()

        xt0 = torch.cat((torch.zeros_like(t), t), 1)
        xt1 = torch.cat((self.space_bounds[0] * torch.ones_like(t), t), 1)

        boundary_loss = (self.forward(xt0)).square().mean() + (self.forward(xt1)).square().mean()

        x0 = torch.cat((x, torch.zeros_like(x)), 1)

        initial_loss = (self.forward(x0) - self.f(x)).square().mean()

        loss_value = loss + boundary_loss + initial_loss
        return loss_value

    def test_loss(self, data):
        x = torch.tensor(data[:, 0:1]).to(self.dev)
        y_t_exact = self.final_sol(x.t())
        x_t = torch.cat((x, self.T * torch.ones_like(x)), 1)
        y_t_net = torch.squeeze(self.forward(x_t))
        l2_err = (y_t_exact - y_t_net).square().mean()
        ref = y_t_exact.square().mean()

        return (l2_err / ref).sqrt()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
