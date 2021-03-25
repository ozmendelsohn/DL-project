from data_utils import *
from toy_model import *
from test_utils import *
from train_utils import *
import pandas as pd
import torch
import argparse
n_atoms = read('datasets/40-cspbbr3-795K.xyz').get_global_number_of_atoms()

# %% argparse setup
parser = argparse.ArgumentParser(description='Input information for the training')
parser.add_argument('--hidden_size', type=int, default=32,
                    help='')
parser.add_argument('--repeat', type=int, default=2,
                    help='')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='')
parser.add_argument('--weight_decay', type=float, default=1,
                    help='')
parser.add_argument('--epochs', type=int, default=100,
                    help='')
parser.add_argument('--batch_size', type=int, default=100,  # 100GB!
                    help='')
parser.add_argument('--test_batch_size', type=int, default=100,  # 100GB!
                    help='')

parser.add_argument('--init_state_path', type=str, default='initialize_model.torch',
                    help='')
parser.add_argument('--ReInitialized_MODEL', type=bool, default=True,
                    help='')
parser.add_argument('--save_model', type=bool, default=True,
                    help='')
parser.add_argument('--model_path', type=str, default='',
                    help='')
parser.add_argument('--search_method', type=str, default='',
                    help='The name of the single run')
parser.add_argument('--name', type=str, default='namey-macnameface',
                    help='The name of the single run')
parser.add_argument('--project', type=str, default='sGDML-train',
                    help='The name of the wandb project')
args = parser.parse_args()
ReInitialized_MODEL = args.ReInitialized_MODEL
init_state_path = args.init_state_path
hidden_size = args.hidden_size
repeat = args.repeat
lr = args.lr
weight_decay = args.weight_decay
epochs = args.epochs
batch_size = args.batch_size
test_batch_size = args.test_batch_size
name = args.name
project = args.project

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
r2, a, b, RMSE, MSE, val_loss, name = \
    train_over_hyperparameters(model, init_state_path,
                               trainset, testset,
                               criteria,
                               lr=lr, weight_decay=weight_decay, epochs=epochs,
                               batch_size=batch_size, test_batch_size=test_batch_size,
                               name=name,
                               cuda=True, parallel=True, verbose=True)
if args.save_model:
    torch.save(model.state_dict(), args.model_path + '/' + name + '.model')

# %% Logging run
LOGGING_FILE = 'bookkeeping.json'

line = dict(name=name, lr=lr, weight_decay=weight_decay, epochs=epochs,
            batch_size=batch_size, test_batch_size=test_batch_size,
            hidden_size=hidden_size, repeat=repeat,
            search_method=args.search_method,
            R2=r2, slope=a, intercept=b, RMSE=RMSE, MSE=MSE, validation_loss=val_loss)
try:
    df = pd.read_json(LOGGING_FILE)
except ValueError:
    df = pd.DataFrame()
df = df.append(line, ignore_index=True)
df.to_json(LOGGING_FILE)

