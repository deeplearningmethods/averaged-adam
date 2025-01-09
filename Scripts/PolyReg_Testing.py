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

deg = 25
n = 50000
sigma = 0.2


def P(x):  # function to be approximated
    # return x ** 4 - 3. * x ** 2 + 1.
    # return x ** 2 - x + 1.
    return torch.sin(math.pi * x)


ann = PolyReg1d(deg, P, n, dev, sigma)
averaged_ann = PolyReg1d(deg, P, n, dev, sigma)

train_steps = 200000
eval_steps = train_steps // 200

M = 1000
factor_list = [1]
gamma_list = [0.99, 0.999]

bs = 64
lr = 0.01
lr_exp = None

results = test_sgd_av_adam(ann, train_steps, eval_steps, bs, n, lr, gamma_list, factor_list, M, lr_exp=lr_exp, with_sgd=True)

json_path = f'PolyReg_{deg}d_AveragedAdam.json'
with open(json_path, 'w') as f:
    json.dump(results, f)
