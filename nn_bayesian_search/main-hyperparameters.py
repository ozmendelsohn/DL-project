import sys
from matplotlib import use
use('Agg')

sys.path.append('..')
from data_utils import *
from toy_model import *
from test_utils import *
from train_utils import *
from matplotlib import pyplot as plt
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
import petname
import pandas as pd
import torch
import argparse

n_atoms = read('../datasets/40-cspbbr3-795K.xyz').get_global_number_of_atoms()

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
parser.add_argument('--batch_size', type=int, default=100,
                    help='')
parser.add_argument('--test_batch_size', type=int, default=100,
                    help='')

parser.add_argument('--init_state_path', type=str, default='initialize_model.torch',
                    help='')
parser.add_argument('--ReInitialized_MODEL', type=bool, default=True,
                    help='')
parser.add_argument('--save_model', type=bool, default=True,
                    help='')
parser.add_argument('--model_path', type=str, default='../logs',
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
    for file in os.listdir('../datasets')[:]:
        if '300K' in file:
            continue
        else:
            training_datasets.append('../datasets/' + file)
    trainset = xyzDatasetTensor(training_datasets)
    testset = xyzDatasetTensor('../datasets/40-cspbbr3-300K.xyz')
    torch.save(trainset, 'trainset.torch')
    torch.save(testset, 'testset.torch')
else:
    trainset = torch.load('trainset.torch')
    testset = torch.load('testset.torch')
print(f'number of traning points: {len(trainset)}')


# %% Model Setup
def model_setup_run(setup=(hidden_size, repeat, lr, weight_decay, batch_size),
                    epochs=epochs,
                    n_atoms=n_atoms, criteria=nn.SmoothL1Loss(),
                    trainset=trainset, testset=testset,
                    test_batch_size=test_batch_size,
                    name=name,
                    init_state_path=init_state_path, ReInitialized_MODEL=ReInitialized_MODEL):
    name = petname.Generate(3, ' ', 10)
    hidden_size, repeat, lr, weight_decay, batch_size = setup
    module = nn.Linear(hidden_size, hidden_size)
    batch_size = int(batch_size)
    input_module = nn.Linear(n_atoms * 3, hidden_size)
    output_module = nn.Linear(hidden_size, n_atoms * 3)
    model = modular_nn(input_module, module, repeat, output_module)
    if ReInitialized_MODEL:
        torch.save(model.state_dict(), init_state_path)

    print(model)
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
    return RMSE


# %%
def res_callback(res):
    fig, axs = plt.subplots(nrows=2, gridspec_kw=dict(hspace=0.3))
    # axs[0].set_xscale('log')
    sigs = [x[0] for x in res.x_iters]
    rmse = res.func_vals
    axs[0].scatter(sigs[-1], rmse[-1], s=300, color='C8')
    sigs, rmse = zip(*sorted(zip(sigs, rmse)))
    axs[0].plot(sigs, rmse, ':*')
    # axs[0].set_xscale('log')
    plot_convergence(res, ax=axs[1])

    title = f'f_RMSE: {res.fun:.5f}'
    axs[0].set_title(title)
    fig.savefig(f'convergence.png', dpi=200)
    plt.close('all')


#
res = gp_minimize(model_setup_run,  # the function to minimize
                  [Integer(low=12, high=256, name='hidden_size', prior='log-uniform'),
                   Integer(low=1, high=5, name='repeat', prior='uniform'),
                   Real(high=1e-2, low=1e-8, name='lr', prior='log-uniform'),
                   Real(high=1, low=0.5, name='weight_decay', prior='log-uniform'),
                   Integer(low=100, high=1000, name='batch_size', prior='log-uniform'),
                   ],  # the bounds on each dimension of x
                  acq_func="EI",  # the acquisition function
                  n_calls=20,  # the number of evaluations of f
                  n_initial_points=3,  # the number of random initialization points
                  verbose=True,
                  callback=res_callback,
                  noise=0.00001 ** 2,  # the noise level (optional)
                  random_state=1234)  # the random seed

print("x^*=%.4f, f(x^*)=%.4f" % (res.x[0], res.fun))
