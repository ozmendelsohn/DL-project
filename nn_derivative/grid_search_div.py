import sys
sys.path.append('..')
# from data_utils import *
# from toy_model import *
# from test_utils import *
from train_utils import *
import sys
from itertools import product
from subprocess import call
import argparse

parser = argparse.ArgumentParser(description='Input information for the run')
parser.add_argument('--queue', type=str, default='gpu-short',
                    help='')
sys.path.append('/home/labs/lkronik/ozyosem/sync/')
from lsf_utils import lsf_utils

work_dir = '/home/labs/lkronik/ozyosem/sync/work/sgdml/cspbbr3'
args = parser.parse_args()
queue = args.queue
core, gpu, mem = 2, 1, '50GB'
container = 'ibdgx001:5000/schnet0.4.0rc:15-3-2021'

ReInitialized_MODEL = False
init_state_path = 'initialize_model.torch'
hidden_size = 128
repeat = 5

hidden_size = [hidden_size]
repeat = [repeat]
n_lr, n_wd, n_rho = 30, 30, 10
learning_rate = np.logspace(-2, -8, n_lr)
weight_decay = np.logspace(-0.00, -0.1, n_wd)
rho_tradeoff = np.logspace(-1, -10, n_rho)
epochs = [100]
batch_size = [1000]
test_batch_size = [100]

project = "cspbbr3-sgdml-train"
path = work_dir + project + '/'
execute_lines, lsf_kwargs, parser_kwargs = [], [], []
for lr, wd, rho, e, bs, bst, hs, rep in tqdm(product(
        learning_rate, weight_decay, rho_tradeoff, epochs, batch_size, test_batch_size, hidden_size, repeat),
                               total=n_lr * n_wd):
    name = lsf_utils.unique_name(3, 4)
    execute_lines.append(
        f'ulimit -Sc 1 \n'
        # f'pip install scikit-learn\n' +
        'python -u main-derivative.py')
    # 'python -u train-script.py ')
    lsf_kwargs.append(dict(mail='ozyosef.mendelsohn@weizmann.ac.il',
                           name=name,
                           core=core,
                           mem=mem,
                           gpu=gpu,
                           container=container,
                           queue=queue,
                           path=path))
    parser_kwargs.append(dict(lr=lr,
                              weight_decay=wd,
                              epochs=e,
                              batch_size=bs,
                              test_batch_size=bst,
                              hidden_size=hs,
                              repeat=rep,
                              name=name,
                              ReInitialized_MODEL=ReInitialized_MODEL,
                              init_state_path=init_state_path,
                              model_path='logs',
                              search_method='gird'))

job_file = f'bookkeeping-{queue}'
try:
    job_df = pd.read_json(f'{job_file}.json')
except ValueError:
    module = nn.Sequential(nn.Linear(hidden_size[0], hidden_size[0]), nn.ReLU())
    input_module = nn.Sequential(nn.Linear(n_atoms * 3, hidden_size[0]), nn.ReLU())
    output_module = nn.Linear(hidden_size[0], n_atoms * 3)
    model = modular_nn(input_module, module, repeat[0], output_module)
    torch.save(model.state_dict(), init_state_path)
    job_df = lsf_utils.jobs_dataframe(execute_lines, lsf_kwargs, parser_kwargs)
lsf_utils.submit_over_df(job_df, max_time=744, wait_time=1, max_queue=1, restart=3,
                         dump=job_file)
import os

os.remove(f'{job_file}.json')
os.remove(f'{job_file}.csv')
print('------------job done!-------------')
