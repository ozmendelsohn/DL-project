from data_utils import *
from toy_model import *
from test_utils import *
from train_utils import *
import pandas as pd
import torch
n_atoms = read('datasets/40-cspbbr3-795K.xyz').get_global_number_of_atoms()

# %% load data
RELOAD_DATA = False

if RELOAD_DATA:
    training_datasets = []
    for file in os.listdir('datasets')[:2]:
        if '300K' in file:
            continue
        else:
            training_datasets.append('datasets/' + file)
    trainset = xyzDatasetTensor(training_datasets, target='energy')
    testset = xyzDatasetTensor('datasets/40-cspbbr3-300K.xyz', target='energy')
    torch.save(trainset, 'trainset-energy.torch')
    torch.save(testset, 'trainset-energy.torch')
else:
    trainset = torch.load('trainset-energy.torch')
    testset = torch.load('trainset-energy.torch')
print(f'number of traning points: {len(trainset)}')


# %% Model Setup
hidden_size = 8
repeat = 3

module = nn.Linear(hidden_size, hidden_size)
input_module = nn.Linear(n_atoms * 3, hidden_size)
output_module = nn.Linear(hidden_size, 1)
model = modular_nn(input_module, module, repeat, output_module)
