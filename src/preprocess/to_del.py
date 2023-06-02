#%% Import packages
from pathlib import Path 
import os, sys
import zipfile
import shutil
import sys
import numpy as np

#%%
#%% extract params and file locations from snakemake
# fn_out = snakemake.output
fn_in = "/p/11208719-interreg/data/racmo/members_bias_corrected/a_raw/hourly/data.zip"

#extracting params
member_number = 'r10i1p5f1'
main_folder = 'full_ds'
year_name = '2014'
ext = 'h'
#year_end = snakemake.params.year_end
var_name = "t2m" #pet, t2m
dt_name = "hourly"
fn_extract = "/p/11208719-interreg/data/racmo/members_bias_corrected/a_raw"

fn_extract = os.path.join(fn_extract, dt_name)

#%%
#Inital file location
for year in np.arange(1950,2015):
    year_name = f'{int(year)}'
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

#%% cdo_regrid
import os, glob
from cdo import *
import sys
#%%

#Initialize cdo
cdo = Cdo()

#We parse the snakemake params

# Folder_src = snakemake.params.f_src
# Folder_dst = snakemake.params.f_dst
grid = '/p/11208719-interreg/wflow/wflow_meuse_julia/inmaps_racmo/wflow_fews_meuse_grid.txt'
variable = "precip" #pet, t2m, precip
dt = "hourly"

files = glob.glob(os.path.join('/p/11208719-interreg/data/racmo/members_bias_corrected/a_raw/hourly/full_ds/r10i1p5f1', f'{variable}', '*.nc'))

for file in files:
    print("Doing file: ", file)
    fn_name = file.split(f'/{variable}/')[-1].split('.nc')[0]

    if any( str(year) in fn_name for year in np.arange(1990,1996)):
        print(fn_name)

        print("fn_name is ", fn_name)
        outfile_path = os.path.join('/p/11208719-interreg/data/racmo/members_bias_corrected/b_preprocess/hourly', f'{variable}', fn_name+'_regrid_meuse.nc')
        print("outfile_path is ", outfile_path)

        if variable == "t2m":
            cdo.remapnn(grid, input='-selvar,t2m {}'.format(str(file)), output=outfile_path, options = "-f nc") 

        if variable == "precip":
            cdo.remapnn(grid, input='-setrtoc,-999.0,0,0  -selvar,precip {}'.format(str(file)), output=outfile_path, options = "-f nc") 
        #            cdo.setrtoc(input='-999.0,0,0 -remapnn,{} -selvar,precip {}'.format(grid, file), output=outfile_path, options = "-f nc") 

        if variable == "pet":
            cdo.remapnn(grid, input='-setrtoc,-999.0,0,0 {}'.format(str(file)), output=outfile_path, options = "-f nc") 

        print("File done")  
    else:
        continue

print("Done for timestep {} and variable {}".format(dt, variable))

#%% merge
import xarray as xr
import os, glob, sys

conv_params={'pet':{
                        'daily': 86400,
                        'hourly': 3600,
                        'conversion': "multiplicative",
                        'expected_original_units': ['kg m-2 s-1','kg/m2/s'] ,#Attributes of the units in the cdo netcdf file
                        'new_units': 'mm/timestep',
                        'wflow_name': "pet"},
            'precip': {
                        'daily': 86400,
                        'hourly': 3600,
                        'conversion': "multiplicative",
                        'expected_original_units': ['kg m-2 s-1','kg/m2/s'],
                        'new_units': 'mm/timestep',
                        'wflow_name': "precip"},
            't2m':{
                        'daily': -273.15,
                        'hourly': -273.15,
                        'conversion': "additive",
                        'expected_original_units': ['K'],
                        'new_units': 'C',
                        'wflow_name': "temp"}}

year = 1990
member_nb = 'r10i1p5f1'

file_temp = "/p/11208719-interreg/data/racmo/members_bias_corrected/b_preprocess/hourly/"+"t2m"+f"/t2m.KNMI-{year}.{member_nb}_regrid_meuse"+".nc"
file_pet = "/p/11208719-interreg/data/racmo/members_bias_corrected/b_preprocess/hourly/"+"pet"+f"/pet.KNMI-{year}.{member_nb}_regrid_meuse"+".nc"
file_precip = "/p/11208719-interreg/data/racmo/members_bias_corrected/b_preprocess/hourly/"+"precip"+f"/precip.KNMI-{year}.{member_nb}_regrid_meuse"+".nc"

outfile_path = f"/p/11208719-interreg/data/racmo/members_bias_corrected/c_wflow/hourly/ds_merged_{year}.nc"
path_dict={'pet': file_pet,  #Check if this works
            't2m': file_temp,
            'precip': file_precip}

#We open the datasets
ds_merge = xr.open_mfdataset([file_temp, file_pet, file_precip], chunks='auto')
del ds_merge.attrs['units']

if 'height' in ds_merge.coords.keys():
    ds_merge = ds_merge.squeeze().drop_vars('height') #We remove the height dimension - not needed here

#Adding the attribute to the variable instead
for var in ds_merge.data_vars:
    print(var)
    ds_var = xr.open_dataset(path_dict[var], chunks='auto')
    ds_merge[var].attrs = ds_var.attrs
    ds_var.close()

check_units(ds_merge, conv_params)
convert_to_wflow_units(ds_merge, conv_params, 'hourly')

#We rename the variables in accordance to wflow standards
for var in ds_merge.data_vars:
    print(var)
    ds_merge = ds_merge.rename({var: conv_params[var]['wflow_name']})

#encoding otherwise wflow doesn't run
chunksizes = (1, ds_merge.lat.size, ds_merge.lon.size)
encoding = {
        v: {"zlib": True, "dtype": "float32", "chunksizes": chunksizes}
        for v in ds_merge.data_vars.keys()
    }
encoding["time"] = {"_FillValue": None}

#We save the output
#fn_out = os.path.join(f"{f_wflow_input}/{dt}/{member_nb}/ds_merged_{year_name}.nc")
ds_merge.to_netcdf(outfile_path, encoding=encoding)


