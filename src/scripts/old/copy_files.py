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
fn_in = str(snakemake.input)
fn_out = str(snakemake.output)

# fn= snakemake.params.fn
# sample_name = snakemake.params.samples_from_wc

print('Member: {}'.format(fn_out))

shutil.copyfile(os.path.join(fn_in), os.path.join(fn_out))
# p = Path(os.path.join(fn, f'{sample_name}.csv'))
# print(p)
# p.rename(p.with_suffix('.tsv'))

print("Files copied")
