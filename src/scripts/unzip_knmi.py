#%% Import packages
from pathlib import Path 
import os, sys
import zipfile
import shutil
import sys
import numpy as np

#%% extract params and file locations from snakemake
fn_out = snakemake.output
fn_in = snakemake.input.f_zip

#extracting params
member_number = snakemake.params.member_number
#member_name = snakemake.params.member_name
year_name = snakemake.params.year_name
#year_start = snakemake.params.year_start
#year_end = snakemake.params.year_end
var_name = snakemake.params.var_name
dt_name = snakemake.params.dt_name
fn_extract = snakemake.params.extract_to

fn_extract = os.path.join(fn_extract, dt_name)

print("------- Checking what we got ------")
print("Member number is: ", member_number)
#print("Member name is: ", member_name)
print("Year name is: ", year_name)
#print("Year name is: ", year_start)
#print("Year name is: ", year_end)
print("Var name is: ", var_name)
print("extract location: ", fn_extract)


#Extracting to correct location:
# fn_extract = "../../../data/racmo/members_bias_corrected/a_raw/daily"

#Inital file location
file = os.path.join(f"full_ds/{member_number}/{var_name}/{var_name}.KNMI-{year_name}.KEXT12.kR2v3-v578-fECEARTH3-ds23-{member_number}+hist.DD.nc")
print("File to extract: ", file)
with zipfile.ZipFile(os.path.join(fn_in), "r") as zip_ref:
    zip_ref.extract(file, path=fn_extract)   

#We rename the file for simplicity later in the chain
new_name = os.path.join(f"full_ds/{member_number}/{var_name}/{var_name}.KNMI-{year_name}.{member_number}.nc")
os.rename(os.path.join(fn_extract, file), os.path.join(fn_extract, new_name))
print("File renamed to: ", new_name)

print("Done!")             

