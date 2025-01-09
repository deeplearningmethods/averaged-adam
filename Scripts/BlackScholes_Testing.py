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

d_i = 20
neurons = [d_i, 50, 100, 50, 1]

T = 1.

cor_noise = True
if cor_noise:  # example with correlated noise
    r, c, K = 0.05, 0.1, 110.
    Q = np.ones([d_i, d_i]) * 0.5
    np.fill_diagonal(Q, 1.)
    sigma = torch.Tensor(0.1 + 0.5 * np.linspace(start=1. / d_i, stop=1., num=d_i, endpoint=True))


    def phi(x):  # initial value of solution of heat PDE
        return np.exp(-r * T) * torch.maximum(K - torch.min(x, dim=-1, keepdim=True)[0], torch.tensor(0.))

else:  # example with independent geometric Brownian motions
    r, c, K = 0.05, 0.1, 100.
    Q = np.eye(d_i)
    sigma = torch.linspace(start=1. / d_i, end=.5, steps=d_i)


    def phi(x):  # initial value of solution of heat PDE
        return np.exp(-r * T) * torch.maximum(torch.max(x, dim=-1, keepdim=True)[0] - K, torch.tensor(0.))

space_bounds = [90, 110]

n_test = 4096
mc_samples = 1024
mc_rounds = 200

train_steps = 100000
eval_steps = train_steps // 100

decr_lr = False
if decr_lr:
    lr = 0.005
    lr_exp = 0.25
else:
    lr = .0005
    lr_exp = None

bs = 2048

M = 1000
factor_list = [1]
gamma_list = [0.99, 0.999]

ann = BlackScholes_ANN(neurons, phi, space_bounds, T, c, r, Q, sigma, dev, activation, lr,
                       mc_samples=mc_samples, mc_rounds=mc_rounds, test_size=n_test).to(dev)

results = test_sgd_av_adam(ann, train_steps, eval_steps, bs, n_test, lr, gamma_list, factor_list, M,
                           lr_exp=lr_exp)

json_path = f'BlackScholes_CorNoise{cor_noise}_DecrLr{decr_lr}_AveragedAdam.json'
with open(json_path, 'w') as f:
    json.dump(results, f)
