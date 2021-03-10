import os
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import scipy as sp
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

class xyzDatasetTensor(Dataset):
    def __init__(self, xyz_files, target='forces'):
        """
        """
        if isinstance(xyz_files, str):
            xyz_files = [xyz_files]
        x, y = [], []
        self.n_samples = 0
        for xyz_file in xyz_files:
            molecules = io.read(xyz_file, index=':')
            molecules = [mol for mol in molecules if
                         mol.get_calculator() is not None]  # filter incomplete outputs from trajectory
            lattice = np.array(molecules[0].get_cell())
            Z = np.array([mol.get_atomic_numbers() for mol in molecules])
            all_z_the_same = (Z == Z[0]).all()
            if not all_z_the_same:
                print('Order of atoms changes accross dataset!\n Bye!')
                exit()
            z = Z[0]
            self.n_samples += len(molecules)

            for index, mol in tqdm(zip(range(len(molecules)), molecules), total=len(molecules)):
                xi = mol.positions.reshape([-1, 3*mol.get_global_number_of_atoms()])
                if target == 'forces':
                    yi = mol.get_forces().reshape([-1, 3*mol.get_global_number_of_atoms()])
                elif target == 'energy':
                    yi = mol.get_potential_energy()

                x.append(torch.Tensor(xi))
                y.append(torch.Tensor(yi))
        self.n_samples = len(x)
        self.x = torch.stack(x)
        self.y = torch.stack(y)


    def __getitem__(self, index):
        x = self.x[index, :]
        y = self.y[index, :]
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


def pbc_d(diffs, lat):
    """
    Clamp differences of vectors to super cell.

    Parameters
    ----------
        diffs : :obj:`numpy.ndarray`
            N x 3 matrix of N pairwise differences between vectors `u - v`
        lat_and_inv : tuple of :obj:`numpy.ndarray`
            Tuple of 3 x 3 matrix containing lattice vectors as columns and its inverse.

    Returns
    -------
        :obj:`numpy.ndarray`
            N x 3 matrix clamped differences
    """
    lat_inv = np.linalg.inv(lat)

    c = lat_inv.dot(diffs.T)
    diffs -= lat.dot(np.rint(c)).T

    return diffs


def dist(r, lat=None):
    """
    Compute pairwise Euclidean distance matrix between all atoms.

    Parameters
    ----------
        r : :obj:`numpy.ndarray`
            Array of size 3N containing the Cartesian coordinates of
            each atom.
        lat_and_inv : tuple of :obj:`numpy.ndarray`, optional
            Tuple of 3x3 matrix containing lattice vectors as columns and its inverse.

    Returns
    -------
        :obj:`numpy.ndarray`
            Array of size N x N containing all pairwise distances between atoms.
    """

    n_atoms = r.shape[0]

    if lat is None:
        pdist = sp.spatial.distance.pdist(r, 'euclidean')
    else:
        pdist = sp.spatial.distance.pdist(
            r, lambda u, v: np.linalg.norm(pbc_d(u - v, lat))
        )

    tril_idxs = np.tril_indices(n_atoms, k=-1)
    return sp.spatial.distance.squareform(pdist, checks=False)[tril_idxs]


def dist_mat(r, lat=None):
    """
    Compute pairwise Euclidean distance matrix between all atoms.

    Parameters
    ----------
        r : :obj:`numpy.ndarray`
            Array of size 3N containing the Cartesian coordinates of
            each atom.
        lat_and_inv : tuple of :obj:`numpy.ndarray`, optional
            Tuple of 3x3 matrix containing lattice vectors as columns and its inverse.

    Returns
    -------
        :obj:`numpy.ndarray`
            Array of size N x N containing all pairwise distances between atoms.
    """

    if lat is None:
        pdist = sp.spatial.distance.pdist(r, 'euclidean')
    else:
        pdist = sp.spatial.distance.pdist(
            r, lambda u, v: np.linalg.norm(pbc_d(u - v, lat))
        )

    return sp.spatial.distance.squareform(pdist, checks=False)


def x_to_d(x, lat=None):
    x = x.reshape([-1, 3])
    n_atoms = x.shape[0]
    d = dist_mat(x, lat=lat)
    d = d[np.triu_indices(n_atoms, k=1)]
    return d


class distanceDatasetTensor(Dataset):
    def __init__(self, xyz_files, target='energy'):
        """
        :type df: DataFrame
        """
        if isinstance(xyz_files, str):
            xyz_files = [xyz_files]
        x, y = [], []
        self.n_samples = 0
        for xyz_file in xyz_files:
            molecules = io.read(xyz_file, index=':')
            molecules = [mol for mol in molecules if
                         mol.get_calculator() is not None]  # filter incomplete outputs from trajectory
            lattice = np.array(molecules[0].get_cell())
            Z = np.array([mol.get_atomic_numbers() for mol in molecules])
            all_z_the_same = (Z == Z[0]).all()
            if not all_z_the_same:
                print('Order of atoms changes accross dataset!\n Bye!')
                exit()
            z = Z[0]
            self.n_samples += len(molecules)

            for index, mol in tqdm(zip(range(len(molecules)), molecules), total=len(molecules)):
                xi = x_to_d(mol.positions, lat=lattice)
                yi = mol.get_forces()
                x.append(torch.Tensor(xi))
                y.append(torch.Tensor(yi))
        self.n_samples = len(x)
        self.x = torch.stack(x)
        self.y = torch.stack(y)


    def __getitem__(self, index):
        x = self.x[index, :]
        y = self.y[index, :]
        return x, y

    def __len__(self):
        return self.n_samples


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
