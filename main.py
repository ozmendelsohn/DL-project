from data_utils import *
from toy_model import *
from scipy import stats
from sklearn.metrics import mean_squared_error
from ase.io import read
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import petname

name = petname.Generate(3, ' ', 10)
print(f'starting: {name}')
writer = SummaryWriter(log_dir=f'runs/{name}')

USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f'Let\'s use {device}!')

n_atoms = read('40-cspbbr3-795K.xyz').get_global_number_of_atoms()

EPOCHS = 100
batch_size = 100
lr = 1e-4
RELOAD_DATA = False

training_datasets = ['datasets/40-cspbbr3-355K.xyz',
                     'datasets/40-cspbbr3-520K.xyz',
                     'datasets/40-cspbbr3-538K.xyz',
                     'datasets/40-cspbbr3-704K.xyz',
                     'datasets/40-cspbbr3-795K.xyz',
                     'datasets/40-cspbbr3-869K.xyz',
                     'datasets/40-cspbbr3-924K.xyz',
                     'datasets/40-cspbbr3-997K.xyz',
                     'datasets/40-cspbbr3-1034K.xyz']
if RELOAD_DATA:
    df_train = pd.DataFrame()
    for td in tqdm(training_datasets):
        df_train = df_train.append(dataframe_from_xyz(td), ignore_index=True)
    df_train.to_json('df_train.json')
else:
    df_train = pd.read_json('df_train.json')
print(f'number of traning points: {len(df_train)}')
# df_train = dataframe_from_xyz('datasets/40-cspbbr3-795K.xyz')
trainset = torch.utils.data.DataLoader(xyzDataset(df_train, target='energy'), batch_size=batch_size, shuffle=True)
df_test = dataframe_from_xyz('datasets/40-cspbbr3-300K.xyz')
testset = torch.utils.data.DataLoader(xyzDataset(df_test, target='energy'), batch_size=batch_size, shuffle=True)
net = energy_nn(n_atoms * 3, hidden1_size=128, hidden2_size=128)
optimizer = optim.Adam(net.parameters(), lr=lr)
criteria = nn.SmoothL1Loss()  # reduction='sum'
net.to(device)

X = torch.rand(1, n_atoms * 3).to(device)
writer.add_graph(net, X)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    net = nn.DataParallel(net)


def test_loss(nn, testset, criteria):
    with torch.no_grad():
        for i, (X, y) in enumerate(testset):
            X = X.to(device=device).float()
            y = y.to(device=device).float()
            output = nn(X)
            y = y.view(-1, 1)
            val_loss = criteria(y, output)
            break
        return val_loss


def test_model(nn, testset):
    with torch.no_grad():
        for i, (X, y) in enumerate(testset):
            X = X.to(device=device).float()
            y = y.to(device=device).float()
            output = nn(X)
            y = y.view(-1, 1)
            break
        x = [i[0].to(device=torch.device('cpu')).detach().numpy() for i in output]
        t = [i[0].to(device=torch.device('cpu')).detach().numpy() for i in y]
        RMSE = mean_squared_error(x, t, squared=False)
        MSE = mean_squared_error(x, t, squared=True)
        a, b, r2 = get_linregress(x, t)
        return x, t, r2, a, b, RMSE, MSE


def get_linregress(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    if np.isnan(r_value):
        r_value = 0
    if np.isnan(slope):
        slope = 0
    return slope, intercept, r_value ** 2



## Train!
n_iter = 0
for epoch in tqdm(range(EPOCHS)):
    for i, (X, y) in enumerate(trainset):
        X = X.to(device=device)
        y = y.to(device=device)
        output = net(X)
        y = y.view(-1, 1)
        loss = criteria(output, y)
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(rnn.parameters(), 0.1)
        optimizer.step()
        val_loss = test_loss(net, testset, criteria)
        writer.add_scalar('Loss/train', loss, n_iter)
        writer.add_scalar('Loss/val_loss', val_loss, n_iter)
        x, t, r2, a, b, RMSE, MSE = test_model(net, testset)
        writer.add_scalar('Regression/slope', a, n_iter)
        writer.add_scalar('Regression/intercept', b, n_iter)
        writer.add_scalar('Regression/R2', r2, n_iter)
        writer.add_scalar('Regression/RMSE', RMSE, n_iter)
        writer.add_scalar('Regression/MSE', RMSE, n_iter)
        n_iter += 1
    val_loss = test_loss(net, testset, criteria)
    print(f'validation los: {val_loss}')
