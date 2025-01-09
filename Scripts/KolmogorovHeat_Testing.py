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
neurons = [d_i, 50, 100, 50, 1]

T = 2.
rho = 1.


def phi(x):  # initial value of solution of heat PDE
    return x.square().sum(axis=1, keepdim=True)


def u_T(x):  # value of heat PDE solution at final time T
    return x.square().sum(axis=1, keepdim=True) + 2. * rho * T * d_i


space_bounds = [-1, 1]

n_test = 100000

train_steps = 100000
eval_steps = train_steps // 200
bs = 2048
decr_lr = False
if decr_lr:
    lr = 0.005
    lr_exp = 0.25
else:
    lr = .0005
    lr_exp = None

M = 1000
factor_list = [1]
gamma_list = [0.99, 0.999]

ann = Heat_PDE_ANN(neurons, phi, space_bounds, T, rho, dev, activation, lr, u_T).to(dev)

results = test_sgd_av_adam(ann, train_steps, eval_steps, bs, n_test, lr, gamma_list, factor_list, M,
                           lr_exp=lr_exp)

json_path = f'Kolmogorov_DecrLr{decr_lr}_AveragedAdam.json'
with open(json_path, 'w') as f:
    json.dump(results, f)
