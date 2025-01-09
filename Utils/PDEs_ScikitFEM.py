try:
    from skfem import *
    from skfem.helpers import *
    from skfem.models.poisson import laplace, mass
    from scipy.sparse.linalg import splu

except ImportError:
    print('skfem/scipy not installed.')



class FEM_Basis:
    def __init__(self, space_bounds, spacediscr):
        self.space_bounds = space_bounds
        self.nr_spacediscr = spacediscr + 1

        self.element = ElementTriP1()
        self.element_name = self.element.__doc__
        self.mesh, self.basis = create_rectangular_basis(space_bounds, self.element, spacediscr)
        self.sol_dim = len(self.basis.project(0.))

    def project_cont_function(self, function):
        u0 = self.basis.project(function)

        return u0


class ScikitFEM_PDE:
    def __init__(self, base, diffusivity, nonlin, nonlin_name):
        self.sol_dim = base.sol_dim
        self.space_bounds = base.space_bounds

        self.element = base.element
        self.element_name = base.element_name
        self.mesh = base.mesh
        self.basis = base.basis

        self.nu = diffusivity
        self.nonlin = nonlin
        self.nonlin_name = nonlin_name

        L = - diffusivity * asm(laplace, self.basis)
        M = asm(mass, self.basis)
        self.Stiffness_operator, self.Mass_operator = penalize(L, M, D=self.basis.get_dofs())

    def compute_sol(self, reference_method, initial_values, T_end, nr_timesteps, params):
        u_final = reference_method(T_end, self, initial_values, nr_timesteps, params)

        return u_final


class ReferenceMethod:

    def __init__(self, reference_method, diffusivity, nonlin, nonlin_name, params=None, method_name="No name given"):
        self.reference_method = reference_method

        self.nu = diffusivity
        self.nonlin = nonlin
        self.nonlin_name = nonlin_name
        self.params = params
        self.method_name = method_name

    def create_ode(self, base):
        ode = ScikitFEM_PDE(base, self.nu, self.nonlin, self.nonlin_name)
        return ode

    def compute_sol(self, T, initial_values, nr_timesteps, ode=None, base=None):
        if ode is None:
            ode = self.create_ode(base)

        reference_values = self.reference_method(T, ode, initial_values, nr_timesteps, self.params)

        return reference_values

    def get_reference_name(self):
        return self.method_name + " with params " + str(self.params)


def Second_order_linear_implicit_RK_FEM(T, fem_ode, initial_values, nr_timesteps, params):
    p1 = params[0]
    p2 = params[1]
    timestep_size = float(T) / nr_timesteps
    L = fem_ode.Stiffness_operator
    M = fem_ode.Mass_operator

    apply_solver = splu((M - timestep_size * p2 * L).T).solve

    u = initial_values
    for m in range(nr_timesteps):
        b_one = L @ u + M @ fem_ode.nonlin(u)
        k_one = apply_solver(b_one)

        b_two = L @ (u + 2 * p1 * (0.5 - p2) * timestep_size * k_one) + M @ fem_ode.nonlin(u + timestep_size * p1 * k_one)
        k_two = apply_solver(b_two)

        u = u + timestep_size * ((1 - 1. / (2 * p1)) * k_one + 1. / (2 * p1) * k_two)

    return u


def create_rectangular_basis(space_bounds, element, ncells):

    ref_mesh = MeshTri.init_tensor(np.linspace(0., space_bounds[0], 1 + ncells),
                                   np.linspace(0., space_bounds[1], 1 + ncells))

    basis = Basis(ref_mesh, element)
    return ref_mesh, basis
