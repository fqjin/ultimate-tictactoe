import os
import glob
import shutil
import numpy as np


if os.path.exists('./scramble'):
    os.rmdir('./scramble')
os.mkdir('./scramble')

n = len(glob.glob('./*.npz'))
permute = np.arange(n).reshape((-1, 500)).flatten('F')
# 'F' mode flattens by column, so games are evenly distributed
for i, j in enumerate(permute):
    shutil.copy(f'./{str(j).zfill(5)}.npz''',
                f'./scramble/{str(i).zfill(5)}.npz')
