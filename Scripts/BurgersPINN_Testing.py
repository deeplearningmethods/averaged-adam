from Utils.Algorithms import *
from Utils.ANN_Models import *
import json

torch.set_default_dtype(torch.float64)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Seed = 0

# Set random seeds
np.random.seed(Seed)
torch.manual_seed(Seed)
torch.cuda.manual_seed(Seed)

activation = nn.GELU()

neurons = [2, 16, 32, 16, 1]

space_bounds = [2.]
T = 0.5
alpha = 0.05
beta = 1.1


def init_func(x):
    return 2. * alpha * math.pi * torch.sin(math.pi * x) / (beta + torch.cos(math.pi * x))


def final_sol(x):
    return 2. * alpha * math.pi * torch.sin(math.pi * x) / (beta * math.exp(alpha * T * math.pi ** 2) + torch.cos(math.pi * x))


train_points = 100000
test_points = 20000

ann = Burgers_PINN_1d(neurons, init_func, alpha, space_bounds, final_sol, T, dev, activation=activation,
                      train_points=train_points, test_points=test_points).to(dev)

bs = 128
n_test = 20000

train_steps = 200000
eval_steps = train_steps // 200

lr = 0.003
lr_exp = None
decay = 0

gamma_list = [0.99, 0.999]
M = 1000
factor_list = [1]

results = test_sgd_av_adam(ann, train_steps, eval_steps, bs, n_test, lr, gamma_list, factor_list, M, with_sgd=True,
                           lr_exp=lr_exp, decay=decay)

json_path = 'BurgersPINN_AveragedAdam.json'
with open(json_path, 'w') as f:
    json.dump(results, f)
