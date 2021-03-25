import os
path = '../work/sgdml/cspbbr3/cspbbr3-hightemp-run/'
with open('folders.txt', 'r') as f:
    folders = f.readlines()
    folders = [path + f.strip() for f in folders]

for folder in folders:
    try:
        for x in os.listdir(folder):
            if '.pbs' in x:
                pbs_file = x
    except FileNotFoundError:
        continue

    with open(f'{folder}/{pbs_file}', 'r') as f:
        python_line = f.readlines()[18]
        temp_arg = python_line.split()[6]
        temp = int(temp_arg.split('=')[1])
        im = 0
    for file in os.listdir(f'{folder}/snapshots'):
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
