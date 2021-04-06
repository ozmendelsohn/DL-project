import sys
import numpy as np
from itertools import product
import pandas as pd
import os
import argparse
import getpass

def reset_callback(df, i):
    df.loc[i, 'parser_kwargs']['batch_size'] = int(df.loc[i, 'parser_kwargs']['batch_size']/2)
    return df

parser = argparse.ArgumentParser(description='Input information for the sGDML')
parser.add_argument('--queue', type=str, default='gpu-long',
                    help='')
sys.path.append('/home/labs/lkronik/ozyosem/sync/')
from lsf_utils import lsf_utils

work_dir = '/home/labs/lkronik/ozyosem/sync/work/sgdml/cspbbr3'
args = parser.parse_args()
queue = args.queue
core, gpu, mem = 2, 1, '50GB'
container = 'ibdgx001:5000/schnet0.4.0rc:15-3-2021'

cutoff = [5, 8]  # A
batch_size = [32]
n_features = [64]
n_filters = [64]
n_gaussians = [300]
n_interactions = [6]
SchNet_cutoff = [30]
rho_tradeoff = [0.1, 0.01, 0.001]
learning_rate = [1e-3]
weight_decay = [1, 0.99]
patience = [10]
factor = [0.96]
min_learning_rate = [1e-7]
n_epochs = [2000]
trainable_gaussians = [False, True]

project = "cspbbr3-SchNet-train"
path = work_dir + project + '/'
execute_lines, lsf_kwargs, parser_kwargs = [], [], []

for coff, bs, nfeat, nfilt, ng, ni, g_coff, rho, lr, wd, pt, fctr, min_lr, n_epoch, tg in product(cutoff,
                                                                                             batch_size,
                                                                                             n_features,
                                                                                             n_filters,
                                                                                             n_gaussians,
                                                                                             n_interactions,
                                                                                             SchNet_cutoff,
                                                                                             rho_tradeoff,
                                                                                             learning_rate,
                                                                                             weight_decay,
                                                                                             patience,
                                                                                             factor,
                                                                                             min_learning_rate,
                                                                                             n_epochs,
                                                                                             trainable_gaussians):

    name = lsf_utils.unique_name(3, 4)
    execute_lines.append(
        f'ulimit -Sc 1 \n' +
        'python train.py')
    # 'python -u train-script.py ')
    lsf_kwargs.append(dict(mail='ozyosef.mendelsohn@weizmann.ac.il',
                           name=name,
                           core=core,
                           mem=mem,
                           gpu=gpu,
                           container=container,
                           queue=queue,
                           path=path))

    parser_kwargs.append(dict(cutoff=coff,  # A
                              batch_size=bs,
                              n_features=nfeat,
                              n_filters=nfilt,
                              n_gaussians=ng,
                              n_interactions=ni,
                              SchNet_cutoff=g_coff,
                              rho_tradeoff=rho,
                              lr=lr,
                              weight_decay=wd,
                              patience=pt,
                              factor=fctr,
                              min_lr=min_lr,
                              n_epochs=n_epoch,
                              trainable_gaussians=tg))
    # parser_kwargs.append(dict())

job_file = f'bookkeeping-{queue}'
try:
    job_df = pd.read_json(f'{job_file}.json')
except ValueError:
    job_df = lsf_utils.jobs_dataframe(execute_lines, lsf_kwargs, parser_kwargs)
lsf_utils.submit_over_df(job_df, max_time=744, wait_time=1, max_queue=1, restart=3,
                         dump=job_file, reset_callback=reset_callback)

import os

os.remove(f'{job_file}.json')
os.remove(f'{job_file}.csv')
print('------------job done!-------------')

