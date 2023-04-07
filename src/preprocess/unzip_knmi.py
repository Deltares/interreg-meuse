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
main_folder = snakemake.params.main_folder
year_name = snakemake.params.year_name
ext = snakemake.params.ext
#year_end = snakemake.params.year_end
var_name = snakemake.params.var_name
dt_name = snakemake.params.dt_name
fn_extract = snakemake.params.extract_to

fn_extract = os.path.join(fn_extract, dt_name)

print("------- Checking what we got ------")
print("Member number is: ", member_number)
print("Main folder is: ", main_folder)
print("Year name is: ", year_name)
print("Extension is: ", ext)
#print("Year name is: ", year_end)
print("Var name is: ", var_name)
print("extract location: ", fn_extract)


#Extracting to correct location:
# fn_extract = "../../../data/racmo/members_bias_corrected/a_raw/daily"

#Inital file location
file = os.path.join(f"{main_folder}/{member_number}/{var_name}/{var_name}.KNMI-{year_name}.KEXT12.kR2v3-v578-fECEARTH3-ds23-{member_number}+hist.{ext}.nc")
print("File to extract: ", file)
with zipfile.ZipFile(os.path.join(fn_in), "r") as zip_ref:
    zip_ref.extract(file, path=fn_extract)   

#We rename the file for simplicity later in the chain
new_name = os.path.join(f"{main_folder}/{member_number}/{var_name}/{var_name}.KNMI-{year_name}.{member_number}.nc")
os.rename(os.path.join(fn_extract, file), os.path.join(fn_extract, new_name))
print("File renamed to: ", new_name)

#We rename to uniform location data/full_ds
final_fn = os.path.join(f"full_ds/{member_number}/{var_name}/{var_name}.KNMI-{year_name}.{member_number}.nc")
path = os.path.join(fn_extract, "full_ds")
if not os.path.exists(path):
   os.mkdir(path)
shutil.move(os.path.join(fn_extract, new_name), os.path.join(fn_extract, final_fn))
print("File moved to: ", os.path.join(fn_extract, final_fn))

print("Done!")

