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

neurons = [3, 32, 64, 32, 1]

space_bounds = [2., 1.]


def init_func(x):
    return np.prod(np.sin(np.pi * x), axis=0)


def init_func_square(x):
    return 1.5 * (np.prod(np.sin(np.pi * x), axis=0)) ** 2


def sine_torch(x):
    return torch.sin(x)


def allen_cahn_nonlin(inputs):
    return inputs - inputs * inputs * inputs


def sine_nonlin(inputs):
    return np.sin(inputs)


pde_name = 'AC'

if pde_name == 'AC':  # for Allen-Cahn PDE
    nonlin = allen_cahn_nonlin
    torch_nonlin = allen_cahn_nonlin
    T = 4.
    alpha = 0.01
    f_0 = init_func

elif pde_name == 'SG':  # for Sine-Gordon PDE
    nonlin = sine_nonlin
    torch_nonlin = sine_torch
    T = 1.
    alpha = 0.05
    f_0 = init_func_square

else:
    raise ValueError(f'{pde_name}_PDE has not been implemented.')

train_points = 60000
test_points = 10000

ann = SemilinHeat_PINN_2d(neurons, f_0, nonlin, alpha, space_bounds, T, activation=activation,
                          torch_nonlin=torch_nonlin, nonlin_name=pde_name, train_points=train_points, test_points=test_points)

bs = 256
n_test = 2000

train_steps = 150000
eval_steps = 1000

lr = 0.001
lr_exp = None

gamma_list = [0.99, 0.999]
M = 1000
factor_list = [1]

results = test_sgd_av_adam(ann, train_steps, eval_steps, bs, n_test, lr, gamma_list, factor_list, M, with_sgd=True,
                           lr_exp=lr_exp)

json_path = f'{pde_name}-PINN_AveragedAdam.json'
with open(json_path, 'w') as f:
    json.dump(results, f)
