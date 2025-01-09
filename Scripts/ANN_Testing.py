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

activation_dict = {
    'GELU': nn.GELU(),
    'ReLU': nn.ReLU(),
    'Sigmoid': nn.Sigmoid(),
    'Softplus': nn.Softplus(),
    'Tanh': nn.Tanh()
}

activation = 'ReLU'

target = 'Gauss'

if target == 'Poly':  # polynomial target function
    d_i = 6
    neurons = [d_i, 64, 64, 1]
    lr = 0.01
    eta = 0
    A = torch.reshape(torch.arange(1. - d_i, d_i + 1., 2.), [d_i, 1])

    def f(x):  # target function
        return torch.matmul(x ** 2, A) + 1.
elif target == 'Gauss':  # Gaussian density as target function
    d_i = 20
    neurons = [d_i, 50, 100, 50, 1]
    sigma = 3
    eta = 0.2
    lr = 0.001

    def f(x):  # target function
        return torch.exp(- 0.5 * torch.sum(x ** 2, dim=1, keepdim=True) / sigma ** 2)


train_steps = 100000
eval_steps = train_steps // 200

M = 1000
factor_list = [1]
gamma_list = [0.99, 0.999]

bs = 256
n_test = 20000


ann = Supervised_ANN(neurons, f, [-1., 1.], dev, activation=activation_dict[activation], sigma=eta)

lr_exp = None

results = test_sgd_av_adam(ann, train_steps, eval_steps, bs, n_test, lr, gamma_list, factor_list, M,
                           lr_exp=lr_exp, with_sgd=True)

json_path = f'ANN-{target}_AveragedAdam.json'
with open(json_path, 'w') as f:
    json.dump(results, f)
