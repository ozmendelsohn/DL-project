from data_utils import *
from toy_model import *
from test_utils import *
import numpy as np
from ase.io import read
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import petname
n_atoms = read('datasets/40-cspbbr3-795K.xyz').get_global_number_of_atoms()

# %% Setup
def get_device(USE_GPU=True, verbose=True):
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if verbose:
        print(f'Let\'s use {device}!')
    return device


def train_setup(model, trainset, testset, init_state_path=None,
                batch_size=100,test_batch_size=100,
                lr=1e-4, weight_decay=0.99):
    # %% Training setup
    if init_state_path is not None:
        model.load_state_dict(torch.load(init_state_path))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=test_batch_size, shuffle=True)
    return optimizer, trainloader, testloader


def train(model, optimizer, trainloader, testloader, epochs, criteria, log_writer=None, verbose=False):
    n_iter = 0
    if verbose:
        pbar = tqdm(range(epochs))
    else:
        pbar = range(epochs)
    for _ in pbar:
        for i, (X, y) in enumerate(trainloader):
            X = X.to(device=device).view(-1, n_atoms * 3)
            y = y.to(device=device).view(-1, n_atoms * 3)
            output = model(X)
            loss = criteria(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if log_writer is not None:
                log_writer.add_scalar('Loss/train', loss, n_iter)
            n_iter += 1

        val_loss = test_f_loss(model, testloader, criteria)
        if log_writer is not None:
            x, t, r2, a, b, RMSE, MSE = test_model(model, testloader)
            log_writer.add_scalar('Loss/val_loss', val_loss, n_iter)
            log_writer.add_scalar('Regression/slope', a, n_iter)
            log_writer.add_scalar('Regression/intercept', b, n_iter)
            log_writer.add_scalar('Regression/R2', r2, n_iter)
            log_writer.add_scalar('Regression/RMSE', RMSE, n_iter)
            log_writer.add_scalar('Regression/MSE', RMSE, n_iter)
        # val_loss = test_loss(net, testset, criteria)
        if verbose:
            pbar.set_description(f'validation los: {val_loss:.5f}')


def train_over_hyperparameters(model, init_state_path,
                               trainset, testset,
                               criteria,
                               lr=1e-4, weight_decay=0.99, epochs=100,
                               batch_size=100, test_batch_size=100,
                               name=None,
                               cuda=True, parallel=True, verbose=False):
    name = petname.Generate(3, ' ', 10) if name is None else name
    if verbose:
        print(f'starting: {name}')
    writer = SummaryWriter(log_dir=f'runs/{name}')

    with torch.no_grad():
        X = torch.rand(1, n_atoms * 3)
        writer.add_graph(model, X)

    optimizer, trainloader, testloader = train_setup(model, trainset, testset, init_state_path=init_state_path,
                                                     batch_size=batch_size, test_batch_size=test_batch_size,
                                                     lr=lr, weight_decay=weight_decay)

    device = get_device(cuda, verbose)
    model.eval()
    model.to(device)
    # %% Multi-GPUs
    if torch.cuda.device_count() > 1 and parallel:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    # %% Train!
    train(model, optimizer,  trainloader, testloader, epochs, criteria, writer,  verbose=verbose)
    testloader_all = DataLoader(testset, batch_size=len(testset), shuffle=True)
    val_loss = test_f_loss(model, testloader_all, criteria)
    x, t, r2, a, b, RMSE, MSE = test_model(model, testloader_all)
    writer.add_hparams(dict(lr=lr, weight_decay=weight_decay, epochs=epochs, batch_size=batch_size),
                       {'hparam/R2': r2, 'hparam/slope': a, 'hparam/intercept': b,
                        'hparam/RMSE': RMSE, 'hparam/MSE': MSE, 'hparam/val_loss': val_loss.cpu().numpy()})

    return r2, a, b, RMSE, MSE, val_loss.cpu().numpy().item(), name

