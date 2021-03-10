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
container = 'ibdgx001:5000/schnet:9-3-2021v2'

ReInitialized_MODEL = False
init_state_path = 'initialize_model.torch'
hidden_size = 128
repeat = 5

module = nn.Linear(hidden_size, hidden_size)
input_module = nn.Linear(n_atoms * 3, hidden_size)
output_module = nn.Linear(hidden_size, n_atoms * 3)
model = modular_nn(input_module, module, repeat, output_module)
torch.save(model.state_dict(), init_state_path)

n_lr, n_wd = 30, 30
learning_rate = np.logspace(-2, -8, n_lr)
weight_decay = np.logspace(-0.00, -0.1, n_wd)
epochs = [100]
batch_size = [1000]
test_batch_size = [100]

project = "cspbbr3-sgdml-train"
path = work_dir + project + '/'
execute_lines, lsf_kwargs, parser_kwargs = [], [], []
for lr, wd, e, bs, bst in tqdm(product(learning_rate, weight_decay, epochs, batch_size, test_batch_size),
                               total=n_lr * n_wd):
    name = lsf_utils.unique_name(3, 4)
    execute_lines.append(
        f'ulimit -Sc 1 \n'
        # f'pip install scikit-learn\n' +
        'python main-hyperparameters.py ')
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
                              name=name,
                              ReInitialized_MODEL=False,
                              init_state_path=init_state_path,
                              model_path='logs',
                              search_method='gird'))

job_file = f'bookkeeping-{queue}'
try:
    job_df = pd.read_json(f'{job_file}.json')
except ValueError:
    job_df = lsf_utils.jobs_dataframe(execute_lines, lsf_kwargs, parser_kwargs)
lsf_utils.submit_over_df(job_df, max_time=744, wait_time=1, max_queue=1, restart=3,
                         dump=job_file)
import os

os.remove(f'{job_file}.json')
os.remove(f'{job_file}.csv')
print('------------job done!-------------')