import os
import glob
import shutil
import numpy as np

if os.path.exists('./scramble'):
    shutil.rmtree('./scramble')
os.mkdir('./scramble')

data_range = None
# data_range = (1000, 4000)

if not data_range:
    n = len(glob.glob('./*.npz'))
    print(f'Got {n} games')
    permute = np.arange(n).reshape((-1, 500)).flatten('F')
    for i, j in enumerate(permute):
        shutil.copy(f'./{str(j).zfill(5)}.npz''',
                    f'./scramble/{str(i).zfill(5)}.npz')
else:
    start, stop = data_range
    print(f'Getting games {start} to {stop}')
    permute = np.arange(start, stop).reshape((-1, 500)).flatten('F')
    for i, j in zip(range(start, stop), permute):
        shutil.copy(f'./{str(j).zfill(5)}.npz''',
                    f'./scramble/{str(i).zfill(5)}.npz')
