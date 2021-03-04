from scipy import stats
from sklearn.metrics import mean_squared_error
import torch
import numpy as np
from ase.io import read


USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

n_atoms = read('datasets/40-cspbbr3-795K.xyz').get_global_number_of_atoms()


def test_loss(nn, testset, criteria, device=device):
    with torch.no_grad():
        for i, (X, y) in enumerate(testset):
            X = X.to(device=device).float()
            y = y.to(device=device).float()
            output = nn(X)
            y = y.view(-1, 1)
            val_loss = criteria(y, output)
            break
        return val_loss

def test_f_loss(nn, testset, criteria, device=device, n_atoms=n_atoms):
    with torch.no_grad():
        for i, (X, y) in enumerate(testset):
            X = X.to(device=device).float().view(-1, n_atoms * 3)
            y = y.to(device=device).float().view(-1, n_atoms * 3)
            output = nn(X)
            val_loss = criteria(y, output)
            break
        return val_loss


def test_model(nn, testset, device=device):
    with torch.no_grad():
        for i, (X, y) in enumerate(testset):
            X = X.to(device=device).float()
            y = y.to(device=device).float()
            output = nn(X)
            y = y.view(-1, 1)
            output = output.view(-1, 1)
            break
        x = [i[0].to(device=torch.device('cpu')).detach().numpy() for i in output]
        t = [i[0].to(device=torch.device('cpu')).detach().numpy() for i in y]
        RMSE = mean_squared_error(x, t, squared=False)
        MSE = mean_squared_error(x, t, squared=True)
        a, b, r2 = get_linregress(x, t)
        return x, t, r2, a, b, RMSE, MSE


def get_linregress(x, y, device=device):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    if np.isnan(r_value):
        r_value = 0
    if np.isnan(slope):
        slope = 0
    return slope, intercept, r_value ** 2