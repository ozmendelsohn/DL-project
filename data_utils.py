import os
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from ase import io
from tqdm import tqdm


class xyzDataset(Dataset):
    def __init__(self, df, target='energy'):
        """
        :type df: DataFrame
        """
        x_col, y_col = [], []
        for col in df.columns: 
            if 'position' in col:
                x_col.append(col)
        if target == 'energy':
            y_col.append('E')
        elif target == 'force':
            for col in df.columns:
                if 'force' in col:
                    y_col.append(col)
        self.df = df
        self.x_col = x_col
        self.y_col = y_col
        self.n_samples = len(df)

    def __getitem__(self, index):
        x = torch.Tensor(self.df.loc[index, self.x_col].values)
        y = torch.Tensor(self.df.loc[index, self.y_col].values)
        return x, y

    def __len__(self):
        return self.n_samples

def atoms_dict(atoms):
    d = {}
    for i, (s, pos) in enumerate(zip(atoms.symbols, atoms.positions)):
        d[str(i + 1) + s + '_position_x'] = pos[0]
        d[str(i + 1) + s + '_position_y'] = pos[1]
        d[str(i + 1) + s + '_position_z'] = pos[2]
    try:
        for i, (s, f) in enumerate(zip(atoms.symbols, atoms.get_forces())):
            d[str(i + 1) + s + '_force_x'] = f[0]
            d[str(i + 1) + s + '_force_y'] = f[1]
            d[str(i + 1) + s + '_force_z'] = f[2]
    except RuntimeError:
        pass
    return d

def dataframe_from_xyz(xyz_file):
    molecules = io.read(xyz_file, index=':')
    molecules = [mol for mol in molecules if mol.get_calculator() is not None] # filter incomplete outputs from trajectory
    lattice = np.array(molecules[0].get_cell())
    Z = np.array([mol.get_atomic_numbers() for mol in molecules])
    all_z_the_same = (Z == Z[0]).all()
    if not all_z_the_same:
        print('Order of atoms changes accross dataset!\n Bye!')
        exit()
    z = Z[0]
    df = pd.DataFrame()
    for mol in molecules:
        line_dict = dict(E=mol.get_potential_energy())
        line_dict.update(atoms_dict(mol))
        df = df.append(line_dict, ignore_index=True)
    return df


if __name__ == '__main__':
    path = '../work/sgdml/cspbbr3/cspbbr3-hightemp-run/'
    with open('folders.txt', 'r') as f:
        folders = f.readlines()
        folders = [path + f.strip() for f in folders]

    for folder in folders:
        for x in os.listdir(folder):
            if '.pbs' in x:
                pbs_file = x

        with open(f'{folder}/{pbs_file}', 'r') as f:
            python_line = f.readlines()[18]
            temp_arg = python_line.split()[6]
            temp = int(temp_arg.split('=')[1])
            im = 0
        for file in os.listdir(f'{folder}/snapshots'):
            if 'all' in file:
                continue
            i = int(file.split('-')[1].split('.')[0])
            if i > im:
                im = i
        print(f'max file index: {im}')
        all_snapshots = ''
        for i in range(im + 1):
            with open(f'{folder}/snapshots/cspbbr3-{i}.xyz', 'r') as f:
                all_snapshots += f.read()
        with open(f'datasets/40-cspbbr3-{temp}K.xyz', 'w') as f:
            f.write(all_snapshots)
