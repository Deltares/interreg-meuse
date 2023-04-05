#%% Import packages
from pathlib import Path 
import os
import zipfile
import shutil
import sys
import numpy as np
import pandas as pd

#%% extract params and file locations from snakemake
# SnakeMake arguments are automatically passed WITHOUT importing a lib. So this .py file is basicallly used as a template.
fn_out = str(snakemake.params.o)
print('Member: {}'.format(fn_out))

os.mkdir(fn_out)
#n = np.random.random_integers(low=1, high=10)
n = 5
dataframe = pd.DataFrame(list())

for i in range(0,n):
    print(i)
    dataframe.to_csv(os.path.join(fn_out,f'{i}.csv'))

print("Files created")
