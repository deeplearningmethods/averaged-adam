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

d_i = 25
neurons = [d_i, d_i + 20, d_i + 20, d_i]
input_neurons = [d_i, d_i + 20, d_i + 20, 1]

T = .25
nt = 20


def phi(x):  # terminal value of solution of semilinear PDE
    return torch.log(0.5 * (torch.sum(x.square(), dim=-1) + 1.))


def f(y, v):  # nonlinearity
    return - torch.sum(v.square(), dim=-1)  # Hamilton-Jacobi-Bellman PDE
    # return 0  # heat PDE (no nonlinearity)


bs = 512
n_test = 1
mc_size = 2048
mc_rounds = 400

train_steps = 100000
eval_steps = train_steps // 100
lr = .02
lr_exp = .2

gamma_list = [0.99, 0.999]
factor_list = [1]
M = 1000

batch_normalize = False  # determine if using batch normalization in hidden ANN layers
initial = 'U'
space_bounds = [-1., 1.]

ann = BSDE_Net(neurons, input_neurons, f, phi, T, nt, dev, activation,
                   batch_norm=batch_normalize, mc_rounds=mc_rounds, mc_samples=mc_size, initial=initial, space_bounds=space_bounds).to(dev)

results = test_sgd_av_adam(ann, train_steps, eval_steps, bs, n_test, lr, gamma_list, factor_list, M, with_sgd=True, lr_exp=lr_exp)

json_path = 'BSDE_AveragedAdam.json'
with open(json_path, 'w') as f:
    json.dump(results, f)
