# %%
import wandb
import schnetpack as spk
import os
from schnetpack.environment import AseEnvironmentProvider
from schnetpack.train.hooks import LoggingHook
import argparse

import time
import petname


work_dir = '//home/labs/lkronik/ozyosem/sync/work/sgdml/cspbbr3/'

parser = argparse.ArgumentParser(description='Input information for the sGDML')
parser.add_argument('--cutoff', type=float, default=10,
                    help='')
parser.add_argument('--batch_size', type=int, default=32,
                    help='')
parser.add_argument('--n_features', type=int, default=64,
                    help='')
parser.add_argument('--n_filters', type=int, default=64,
                    help='')
parser.add_argument('--n_gaussians', type=int, default=300,
                    help='')
parser.add_argument('--name', type=str, default='changeme',
                    help='The name of the single run')
parser.add_argument('--n_interactions', type=int, default=6,
                    help='')
parser.add_argument('--SchNet_cutoff', type=float, default=20.,
                    help='')
parser.add_argument('--rho_tradeoff', type=float, default=0.01,
                    help='')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='')
parser.add_argument('--weight_decay', type=float, default=1,
                    help='')
parser.add_argument('--patience', type=int, default=5,
                    help='')
parser.add_argument('--factor', type=float, default=0.96,
                    help='')
parser.add_argument('--min_lr', type=float, default=1e-8,
                    help='')
parser.add_argument('--n_epochs', type=int, default=300,
                    help='')
parser.add_argument('--trainable_gaussians', type=bool, default=False,
                    help='')

args = parser.parse_args()
cutoff = args.cutoff  # A
batch_size = args.batch_size
n_features = args.n_features
n_filters = args.n_filters
n_gaussians = args.n_gaussians
n_interactions = args.n_interactions
SchNet_cutoff = args.SchNet_cutoff
rho_tradeoff = args.rho_tradeoff
lr = args.lr
weight_decay = args.weight_decay
patience = args.patience
factor = args.factor
min_lr = args.min_lr
n_epochs = args.n_epochs
trainable_gaussians = args.trainable_gaussians

run_name = petname.generate(3, letters=10)
wandb.init(project='SchNet-CsPbBr3',
           config=args,
           dir=work_dir,
           name=run_name)
# wandb.init()
# wandb.config.epochs = 4
#
# parser = argparse.ArgumentParser()
# parser.add_argument('-b', '--batch-size', type=int, default=8, metavar='N',
#                     help='input batch size for training (default: 8)')
# args = parser.parse_args()
# wandb.config.update(args)

try:
    os.chdir('SchNet')
except FileNotFoundError:
    pass

forcetut = f'./forcetut/{run_name}'
if not os.path.exists(forcetut):
    os.makedirs(forcetut)


class WandBHook(LoggingHook):
    """Hook for logging training process to CSV files.

    Args:
        log_path (str): path to directory in which log files will be stored.
        metrics (list): metrics to log; each metric has to be a subclass of spk.Metric.
        log_train_loss (bool, optional): enable logging of training loss.
        log_validation_loss (bool, optional): enable logging of validation loss.
        log_learning_rate (bool, optional): enable logging of current learning rate.
        every_n_epochs (int, optional): epochs after which logging takes place.

    """

    def __init__(self,
                 log_path,
                 metrics,
                 log_train_loss=True,
                 log_validation_loss=True,
                 log_learning_rate=True,
                 every_n_epochs=1,
                 ):
        log_path = os.path.join(log_path, "WandBHook.csv")
        super(WandBHook, self).__init__(log_path, metrics, log_train_loss, log_validation_loss, log_learning_rate)
        self._offset = 0
        self._restart = False
        self.every_n_epochs = every_n_epochs

    def on_train_begin(self, trainer):

        pass

    def on_validation_end(self, trainer, val_loss):
        if trainer.epoch % self.every_n_epochs == 0:
            ctime = time.time() + self._offset
            log = str(ctime)
            log_dict = dict()

            if self.log_learning_rate:
                log_dict['learning_rate'] = trainer.optimizer.param_groups[0]["lr"]

            if self.log_train_loss:
                log_dict['train_loss'] = self._train_loss / self._counter

            if self.log_validation_loss:
                log_dict['validation_loss'] = val_loss

            for i, metric in enumerate(self.metrics):
                m = metric.aggregate()
                if hasattr(m, "__iter__"):
                    log_dict[metric.name] = [j for j in m]
                else:
                    log_dict[metric.name] = m

            wandb.log(log_dict)


# %%
# from schnetpack.datasets import MD17
from schnetpack.datasets import AtomsData

cspbbr3_data = AtomsData('40-cspbbr3.db', environment_provider=AseEnvironmentProvider(cutoff=cutoff))

# ethanol_data = MD17(os.path.join(forcetut, 'ethanol.db'), molecule='ethanol')


# for i in range(len(cspbbr3_data)):
example = cspbbr3_data[0]
print(f'Properties of molecule with id {0}:')

for k, v in example.items():
    print('-', k, ':', v.shape)

# example = ethanol_data[0]
print('Properties of molecule with id 0:')

# for k, v in example.items():
#     print('-', k, ':', v.shape)

# %%
atoms, properties = cspbbr3_data.get_properties(0)

# print('Loaded properties:\n', *['{:s}\n'.format(i) for i in properties.keys()])

# %%
# print('Forces:\n', properties['forces'])
# print('Shape:\n', properties['forces'].shape)

# %%
# import matplotlib.pyplot as plt
# from ase.visualize import view
#
# view(atoms, viewer='x3d')
# plt.show()
# %%
train, val, test = spk.train_test_split(
    data=cspbbr3_data,
    num_train=100-000,
    num_val=100-000,
    split_file=os.path.join(forcetut, "split.npz"),
)

train_loader = spk.AtomsLoader(train, batch_size=batch_size, shuffle=True)
val_loader = spk.AtomsLoader(val, batch_size=batch_size)

# print(next(iter(train_loader)))
# %%
means, stddevs = train_loader.get_statistics(
    'energy',  # divide_by_atoms=True
)

print('Mean atomization energy / atom:      {:12.4f} [eV]'.format(means['energy'][0]))
print('Std. dev. atomization energy / atom: {:12.4f} [eV]'.format(stddevs['energy'][0]))

# %%
n_features = 8

schnet = spk.representation.SchNet(
    n_atom_basis=n_features,
    n_filters=n_filters,
    n_gaussians=n_gaussians,
    n_interactions=n_interactions,
    cutoff=SchNet_cutoff,
    cutoff_network=spk.nn.cutoff.CosineCutoff,
    trainable_gaussians=trainable_gaussians,
)

# %%

energy_model = spk.atomistic.Atomwise(
    n_in=n_features,
    property='energy',
    mean=means['energy'],
    stddev=stddevs['energy'],
    derivative='forces',
    negative_dr=True
)

# %%
model = spk.AtomisticModel(representation=schnet, output_modules=energy_model)

# %% Multi-GPUs
import torch
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = torch.nn.DataParallel(model)

# %%
import torch


# loss function the same as https://aip.scitation.org/doi/pdf/10.1063/1.5019779
def loss(batch, result):
    # compute the mean squared error on the energies
    diff_energy = batch['energy'] - result['energy']
    err_sq_energy = torch.mean(diff_energy ** 2)

    # compute the mean squared error on the forces
    diff_forces = batch['forces'] - result['forces']
    err_sq_forces = torch.mean(diff_forces ** 2)

    # build the combined loss function
    err_sq = rho_tradeoff * err_sq_energy + (1 - rho_tradeoff) * err_sq_forces

    return err_sq


# %%
from torch.optim import Adam

# build optimizer
optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# %%
# before setting up the trainer, remove previous training checkpoints and logs
from shutil import rmtree

try:
    rmtree('./forcetut/log.csv')
    rmtree('./forcetut/checkpoints')
except (FileNotFoundError, NotADirectoryError):
    pass

# %%
import schnetpack.train as trn

# set up metrics
metrics = [
    spk.metrics.MeanAbsoluteError('energy'),
    spk.metrics.MeanAbsoluteError('forces'),
    spk.metrics.RootMeanSquaredError('energy'),
    spk.metrics.RootMeanSquaredError('forces'),
]

# construct hooks
hooks = [
    WandBHook(log_path=forcetut, metrics=metrics),
    trn.CSVHook(log_path=forcetut, metrics=metrics),
    trn.ReduceLROnPlateauHook(optimizer,
                              patience=patience, factor=factor, min_lr=min_lr,
                              stop_after_min=True
                              )
]

# %%
trainer = trn.Trainer(
    model_path=forcetut,
    model=model,
    hooks=hooks,
    loss_fn=loss,
    optimizer=optimizer,
    train_loader=train_loader,
    validation_loader=val_loader,
)

# %%
# check if a GPU is available and use a CPU otherwise
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# %%
# determine number of epochs and train

trainer.train(device=device, n_epochs=n_epochs)

# %%
import numpy as np
import matplotlib.pyplot as plt
from ase.units import kcal, mol

# Load logged results
results = np.loadtxt(os.path.join(forcetut, 'log.csv'), skiprows=1, delimiter=',')

# Determine time axis
time = results[:, 0] - results[0, 0]

# Load the validation MAEs
energy_mae = results[:, 4]
forces_mae = results[:, 5]

# Get final validation errors
print('Validation MAE:')
print('    energy: {:10.3f} eV'.format(energy_mae[-1]))
print('    forces: {:10.3f} eV/A'.format(forces_mae[-1]))
wandb.config.e_mae_valid = energy_mae[-1]
wandb.config.f_mae_valid = forces_mae[-1]

#

# %%

best_model = torch.load(os.path.join(forcetut, 'best_model'))

# %%
test_loader = spk.AtomsLoader(test, batch_size=100)

energy_error = 0.0
forces_error = 0.0

for count, batch in enumerate(test_loader):
    # move batch to GPU, if necessary
    batch = {k: v.to(device) for k, v in batch.items()}

    # apply model
    pred = best_model(batch)

    # calculate absolute error of energies
    tmp_energy = torch.sum(torch.abs(pred['energy'] - batch['energy']))
    tmp_energy = tmp_energy.detach().cpu().numpy()  # detach from graph & convert to numpy
    energy_error += tmp_energy

    # calculate absolute error of forces, where we compute the mean over the n_atoms x 3 dimensions
    tmp_forces = torch.sum(
        torch.mean(torch.abs(pred['forces'] - batch['forces']), dim=(1, 2))
    )
    tmp_forces = tmp_forces.detach().cpu().numpy()  # detach from graph & convert to numpy
    forces_error += tmp_forces

    # log progress
    percent = '{:3.2f}'.format(count / len(test_loader) * 100)
    print('Progress:', percent + '%' + ' ' * (5 - len(percent)), end="\r")

energy_error /= len(test)
forces_error /= len(test)

print('\nTest MAE:')
print('    energy: {:10.3f} eV'.format(energy_error))
print('    forces: {:10.3f} eV'.format(forces_error))
wandb.config.e_mae_test = energy_error
wandb.config.f_mae_test = forces_error

# %%
calculator = spk.interfaces.SpkCalculator(
    model=best_model,
    device=device,
    energy='energy',
    forces='forces',
    energy_units='eV',
    forces_units='eV/A'
)

atoms.set_calculator(calculator)

print('Prediction:')
print('energy:', atoms.get_total_energy())
print('forces:', atoms.get_forces())

# %%
# from ase import io
#
# # Generate a directory for the ASE computations
# ase_dir = os.path.join(forcetut, 'ase_calcs')
#
# if not os.path.exists(ase_dir):
#     os.mkdir(ase_dir)
#
# # Write a sample molecule
# molecule_path = os.path.join(ase_dir, 'cspbbr3.xyz')
# io.write(molecule_path, atoms, format='xyz')
#
# # %%
# ethanol_ase = spk.interfaces.AseInterface(
#     molecule_path,
#     best_model,
#     ase_dir,
#     device,
#     energy='energy',
#     forces='forces',
#     energy_units='eV',
#     forces_units='eV/A'
# )
#
# # %%
# ethanol_ase.optimize(fmax=1e-4)
#
# # %%
# ethanol_ase.compute_normal_modes()
#
# # %%
# ethanol_ase.init_md(
#     'simulation'
# )
# # %%
# ethanol_ase.run_md(1000)
# # %%
# # Load logged results
# results = np.loadtxt(os.path.join(ase_dir, 'simulation.log'), skiprows=1)
#
# # Determine time axis
# time = results[:, 0]
#
# # Load energies
# energy_tot = results[:, 1]
# energy_pot = results[:, 2]
# energy_kin = results[:, 3]
#
# # Construct figure
# plt.figure(figsize=(14, 6))
#
# # Plot energies
# plt.subplot(2, 1, 1)
# plt.plot(time, energy_tot, label='Total energy')
# plt.plot(time, energy_pot, label='Potential energy')
# plt.ylabel('E [eV]')
# plt.legend()
#
# plt.subplot(2, 1, 2)
# plt.plot(time, energy_kin, label='Kinetic energy')
# plt.ylabel('E [eV]')
# plt.xlabel('Time [ps]')
# plt.legend()
#
# temperature = results[:, 4]
# print('Average temperature: {:10.2f} K'.format(np.mean(temperature)))
#
# plt.show()
#
# # %%
# ethanol_ase.init_md(
#     'simulation_300K',
#     temp_bath=300,
#     reset=True
# )
# ethanol_ase.run_md(20000)
#
# # %%
# # Load logged results
# results = np.loadtxt(os.path.join(ase_dir, 'simulation_300K.log'), skiprows=1)
#
# # Determine time axis
# time = results[:, 0]
# # 0.02585
# # Load energies
# energy_tot = results[:, 1]
# energy_pot = results[:, 2]
#
# # Construct figure
# plt.figure(figsize=(14, 6))
#
# # Plot energies
# plt.subplot(2, 1, 1)
# plt.plot(time, energy_tot, label='Total energy')
# plt.plot(time, energy_pot, label='Potential energy')
# plt.ylabel('Energies [eV]')
# plt.legend()
#
# # Plot Temperature
# temperature = results[:, 4]
#
# # Compute average temperature
# print('Average temperature: {:10.2f} K'.format(np.mean(temperature)))
#
# plt.subplot(2, 1, 2)
# plt.plot(time, temperature, label='Simulation')
# plt.ylabel('Temperature [K]')
# plt.xlabel('Time [ps]')
# plt.plot(time, np.ones_like(temperature) * 300, label='Target')
# plt.legend()
# plt.show()
