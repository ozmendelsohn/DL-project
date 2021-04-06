from ase.io import read
import numpy as np
from schnetpack import AtomsData
import os
from tqdm import tqdm

# load atoms from xyz file. Here, we only parse the first 10 molecules
atoms = read('./40-cspbbr3-300K.xyz', index=':')
# atoms = read('./ethanol.xyz', index=':10')

# comment line is weirdly stored in the info dictionary as key by ASE. here it corresponds to the energy
print('Energy:', atoms[0].info)
print()

# parse properties as list of dictionaries
property_list = []
for at in tqdm(atoms):
    # All properties need to be stored as numpy arrays.
    # Note: The shape for scalars should be (1,), not ()
    # Note: GPUs work best with float32 data
    # print(at.info.keys())
    # print(list(at.info.keys()))
    # energy = np.array([float(list(at.info.keys())[0])], dtype=np.float32)
    energy = np.array([float(at.info['energy'])], dtype=np.float32)
    forces = np.array(at.get_forces(), dtype=np.float32)
    property_list.append(
        {'energy': energy,
         'forces': forces}
    )
    # print(energy)
    # print(type(energy))

# print('Properties:', property_list)


new_dataset = AtomsData('./40-cspbbr3-300K.db', available_properties=['energy', 'forces'])
new_dataset.add_systems(atoms, property_list)


print('Number of reference calculations:', len(new_dataset))
print('Available properties:')

for p in new_dataset.available_properties:
    print('-', p)
print()

example = new_dataset[0]
print('Properties of molecule with id 0:')

for k, v in example.items():
    print('-', k, ':', v.shape)