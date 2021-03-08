from data_utils import *
from toy_model import *
from test_utils import *
from train_utils import *
import pandas as pd
import torch

# %%
ReInitialized_MODEL = True
init_state_path = 'initialize_model.torch'
# net = force_nn(n_atoms * 3, hidden1_size=16, hidden2_size=16)
hidden_size = 32
repeat = 2
lr = 1e-4
weight_decay = 0.99
epochs = 100
batch_size = 100
test_batch_size = 10

# %% load data
RELOAD_DATA = False

if RELOAD_DATA:
    training_datasets = []
    for file in os.listdir('datasets')[:2]:
        if '300K' in file:
            continue
        else:
            training_datasets.append('datasets/' + file)
    trainset = xyzDatasetTensor(training_datasets)
    testset = xyzDatasetTensor('datasets/40-cspbbr3-300K.xyz')
    torch.save(trainset, 'trainset.torch')
    torch.save(testset, 'testset.torch')
else:
    trainset = torch.load('trainset.torch')
    testset = torch.load('testset.torch')
print(f'number of traning points: {len(trainset)}')

# %% Model Setup

module = nn.Linear(hidden_size, hidden_size)
input_module = nn.Linear(n_atoms * 3, hidden_size)
output_module = nn.Linear(hidden_size, n_atoms * 3)
model = modular_nn(input_module, module, repeat, output_module)
if ReInitialized_MODEL:
    torch.save(model.state_dict(), init_state_path)

print(model)
# %% Train!
criteria = nn.MSELoss()
r2, a, b, RMSE, MSE, val_loss, name = train_over_hyperparameters(model, init_state_path,
                                                                 trainset, testset,
                                                                 criteria,
                                                                 lr=lr, weight_decay=weight_decay, epochs=epochs,
                                                                 batch_size=batch_size, test_batch_size=test_batch_size,
                                                                 cuda=True, parallel=True, verbose=True)

# %% Logging run
LOGGING_FILE = 'bookkeeping.json'

line = dict(name=name, lr=lr, weight_decay=weight_decay, epochs=epochs,
            batch_size=batch_size, test_batch_size=test_batch_size,
            R2=r2, slope=a, intercept=b, RMSE=RMSE, MSE=MSE, validation_loss=val_loss)
try:
    df = pd.read_json(LOGGING_FILE)
except ValueError:
    df = pd.DataFrame()
df = df.append(line, ignore_index=True)
df.to_json(LOGGING_FILE)

