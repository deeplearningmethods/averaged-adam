from Utils.Algorithms import *
from Utils.ANN_Models import *
import json

torch.set_default_dtype(torch.float64)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {dev}')

Seed = 0

# Set random seeds
np.random.seed(Seed)
torch.manual_seed(Seed)
torch.cuda.manual_seed(Seed)

activation = nn.GELU()

d_i = 10
# neurons = [d_i, d_i + 20, d_i + 40, d_i + 20, d_i]
neurons = [d_i, d_i + 30, d_i + 30, d_i]

T = 1.
nt = 100

initial = 'U'  # either N (normal distribution) or U (uniform distribution)
space_bounds = [-1., 1.]

batch_normalize = True  # determine if using batch normalization in hidden ANN layers

bs = 1024
n_test = 4096
mc_size = 400
mc_rounds = 200
mc_test = 500

train_steps = 5000
eval_steps = train_steps // 50
lr = .01
lr_exp = None
lr_decay_start = 0

gamma_list = [0.99, 0.999]
factor_list = [1]
M = 100

sigma = math.sqrt(2)

Q = 0.5
R = 1.
A = np.random.randn(d_i, d_i)
B = np.random.randn(d_i, d_i)
A_torch = torch.Tensor(A.transpose()).to(dev)
B_torch = torch.Tensor(B.transpose()).to(dev)


def phi(x):
    return torch.sum(x.square(), dim=-1)


def f(x, v):  # nonlinear term /running cost
    return Q * torch.sum(x.square(), dim=-1) + R * torch.sum(v.square(), dim=-1)


def mu(x, v):  # drift term
    return torch.matmul(x, A_torch) + torch.matmul(v, B_torch)


ann = ControlNet(neurons, f, phi, mu, T, nt, dev, sigma=sigma, activation=activation, batch_norm=batch_normalize,
                 space_bounds=space_bounds, initial=initial, test_size=n_test, mc_test=mc_test).to(dev)

ode_timesteps = 100000
P, q = solve_ricatti(A, B, Q * np.eye(d_i), R * np.eye(d_i), np.eye(d_i), sigma * np.eye(d_i), T, ode_timesteps)

u_ref = torch.sum(ann.x_test * torch.matmul(ann.x_test, torch.Tensor(P.transpose()).to(dev)), dim=1) + torch.Tensor(
    [q]).to(dev)
ann.u_ref = u_ref.to(dev)

results = test_sgd_av_adam(ann, train_steps, eval_steps, bs, n_test, lr, gamma_list, factor_list, M,
                           with_sgd=True, lr_exp=lr_exp, lr_decay_start=lr_decay_start)

json_path = 'Control_AveragedAdam.json'
with open(json_path, 'w') as f:
    json.dump(results, f)
