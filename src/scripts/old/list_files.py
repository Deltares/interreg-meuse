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

print("The location of the textfile is", fn_out)

#open text file
text_file = open(fn_out, "w")
 
#write string to file
text_file.write(fn_in)
 
#close file
text_file.close()
