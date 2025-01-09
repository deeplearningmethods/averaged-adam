import json

from Utils.Algorithms import *
from Utils.ImageNets import *

import torchvision.datasets as datasets
import torchvision.transforms as transforms


Seed = 0

# Set random seeds
np.random.seed(Seed)
torch.manual_seed(Seed)
torch.cuda.manual_seed(Seed)

torch.set_default_dtype(torch.float64)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {dev}')

ann = ResNet(dev).to(dev)  # define residual network

bs = 64
n_test = 256

transform_dict = {
    'J': transforms.ColorJitter(brightness=0.5),
    'R': transforms.RandomRotation((0, 45)),
    'F': transforms.RandomHorizontalFlip(),
    'C': transforms.RandomCrop(32, padding=4)
}
transform_string = 'FCJR'

transform_train = transforms.Compose([transform_dict[c] for c in transform_string]
                                     + [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))]
                                     )

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
])

cifar_train = datasets.CIFAR10(
    root='./cifar10_data',
    train=True,
    download=True,
    transform=transform_train
)

cifar_test = datasets.CIFAR10(
    root='./cifar10_data',
    train=False,
    download=True,
    transform=transform_test
)

data_loader = torch.utils.data.DataLoader(cifar_train, batch_size=bs, shuffle=True, drop_last=True)
sampler = DataSampler(data_loader)

test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=n_test, shuffle=True, drop_last=True)
test_sampler = DataSampler(test_loader)

train_steps = 80000
eval_steps = 1000
mc_rounds = 10

lr = 0.0002
lr_exp = None
decay = 3e-4

gamma_list = [0.99, 0.999]
M = 1000
factor_list = [1]

results = test_sgd_av_adam(ann, train_steps, eval_steps, bs, n_test, lr, gamma_list, factor_list, M, with_sgd=True,
                           lr_exp=lr_exp, sampler=sampler, test_sampler=test_sampler, with_acc=True,
                           mc_rounds=mc_rounds, decay=decay)

json_path = 'Cifar10_AverageAdam.json'

with open(json_path, 'w') as f:
    json.dump(results, f)
