from data_utils import *
from toy_model import *
from test_utils import *
from ase.io import read
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import petname

# %% Setup
name = petname.Generate(3, ' ', 10)
print(f'starting: {name}')
writer = SummaryWriter(log_dir=f'runs/{name}')

USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f'Let\'s use {device}!')
n_atoms = read('datasets/40-cspbbr3-795K.xyz').get_global_number_of_atoms()


# %% load data
RELOAD_DATA = False

if RELOAD_DATA:
    training_datasets = []
    for file in os.listdir('datasets'):
        if '300K' in file:
            continue
        else:
            training_datasets.append('datasets/' + file)
    # df_train = pd.DataFrame()
    # for td in tqdm(training_datasets):
    #     df_train = df_train.append(dataframe_from_xyz(td), ignore_index=True)
    # trainset = xyzDatasetTensor(df_train, target='force')
    trainset = distanceDatasetTensor(training_datasets)
    # df_test = dataframe_from_xyz('datasets/40-cspbbr3-300K.xyz')
    # testset = xyzDatasetTensor(df_test, target='force')
    testset = distanceDatasetTensor('datasets/40-cspbbr3-300K.xyz')
    # df_train.to_json('df_train.json')
    torch.save(trainset, 'trainset.torch')
    torch.save(testset, 'testset.torch')
else:
    # df_train = pd.read_json('df_train.json')
    trainset = torch.load('trainset.torch')
    # df_test = pd.read_json('df_test.json')
    testset = torch.load('testset.torch')
print(f'number of traning points: {len(trainset)}')


# %% Model Setup
hidden_size = 256
repeat = 10
module = nn.Linear(hidden_size, hidden_size)
input_module = nn.Linear(int(n_atoms*(n_atoms-1)/2), hidden_size)
output_module = nn.Linear(hidden_size, n_atoms * 3)

net = modular_nn(input_module, module, repeat, output_module)
# net = force_nn(n_atoms * 3, hidden1_size=1024, hidden2_size=1024)
print(net)

# %% Training setup
EPOCHS = 100
batch_size = 1000
lr = 1e-6
criteria = nn.SmoothL1Loss()  # reduction='sum'

optimizer = optim.Adam(net.parameters(), lr=lr)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=100, shuffle=True)

net.to(device)
X = torch.rand(1, int(n_atoms*(n_atoms-1)/2)).to(device)
writer.add_graph(net, X)

# %% Multi-GPUs
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    net = nn.DataParallel(net)


#%% Train!
n_iter = 0
# testset = trainset
pbar = tqdm(range(EPOCHS))
for epoch in pbar:
    for i, (X, y) in enumerate(trainloader):
        X = X.to(device=device)
        y = y.to(device=device)
        output = net(X)
        y = y.view(-1, n_atoms * 3)
        loss = criteria(output, y)
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(rnn.parameters(), 0.1)
        optimizer.step()
        val_loss = test_f_loss(net, testloader, criteria)
        writer.add_scalar('Loss/train', loss, n_iter)
        writer.add_scalar('Loss/val_loss', val_loss, n_iter)
        x, t, r2, a, b, RMSE, MSE = test_model(net, testloader)
        writer.add_scalar('Regression/slope', a, n_iter)
        writer.add_scalar('Regression/intercept', b, n_iter)
        writer.add_scalar('Regression/R2', r2, n_iter)
        writer.add_scalar('Regression/RMSE', RMSE, n_iter)
        writer.add_scalar('Regression/MSE', RMSE, n_iter)
        n_iter += 1
    # val_loss = test_loss(net, testset, criteria)
    pbar.set_description(f'validation los: {val_loss}')
