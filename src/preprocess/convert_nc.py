import xarray as xr
import os, glob, sys
#import dask

# We read the snakemake parameters
file_temp = snakemake.input.fn_temp
file_pet = snakemake.input.fn_pet
file_precip = snakemake.input.fn_precip
conv_params = snakemake.params.conv_params
path_dict = snakemake.params.path_dict
year_name = snakemake.params.year_name
dt = snakemake.params.dt_step
outfile_path = str(snakemake.output)

print("------- Checking what we got ------")
print("Outfile path", outfile_path)
print("file_temp", file_temp)
print("file_pet", file_pet)
print("file_precip", file_precip)
print("conv_param", conv_params)
print("path_dict", path_dict)
print("dt", dt)
print("year_name", year_name)

path_dict={'pet': file_pet, 
            't2m': file_temp,
            'precip': file_precip}

#%% Functions 
def check_units(ds, dict_vars):
    '''
    Checking if the units of each variables is what we expect before doing the transformation
    ds: merged netcdf file with the name of the original variables
    dict_vars: dictionnary with as major key the name of the original variable and subkey 'expected_original_units' that mentions the units of the variable
    '''
    for var in ds.data_vars:
        print(var)
        if ds[var].attrs['units'] in dict_vars[var]['expected_original_units']:
            print("units is good")
            continue
        else:
            raise ValueError("Units do not match - check f{var}")

def convert_to_wflow_units(ds, dict_vars, dt):
    '''
    Converting the original units to wflow units as indicated in the snakemake config file and updating the units attributes
    ds: merged netcdf file with the name of the original variables
    dict_vars: dictionnary with as major key the name of the original variable and subkeys indicate the timestep and conversion type (multiplicative or additive)
    '''
    for var in ds.data_vars:
        print(var)
        if dict_vars[var]['conversion'] == 'additive':
            ds[var] = ds[var] + dict_vars[var][dt] #going from Kelvin to C
            ds[var].attrs['units'] = dict_vars[var]['new_units']
            print("additive conversion done")
        elif dict_vars[var]['conversion'] == 'multiplicative':
            ds[var] = ds[var] * dict_vars[var][dt] #going from kg/m2/s to mm/tijdstap
            ds[var].attrs['units'] = dict_vars[var]['new_units']
            print("multiplicative conversion done")
        else:
            print(f"No conversion happened for {var}")
    return ds
        

#%%
#We open the datasets
ds_merge = xr.open_mfdataset([file_temp, file_pet, file_precip], chunks='auto')

if 'height' in ds_merge.coords.keys():
    ds_merge = ds_merge.squeeze().drop_vars('height') #We remove the height dimension - not needed here

#We remove the units attributes from the main attributes as this is confusing
del ds_merge.attrs['units']

#We add each attributes as variables attributes
for var in ds_merge.data_vars:
    print(var)
    ds_var = xr.open_dataset(path_dict[var], chunks='auto')
    ds_merge[var].attrs = ds_var.attrs
    ds_var.close()

#We check the units in the original file
check_units(ds_merge, conv_params)

#We do the conversion according to the values in the snakemake config file
convert_to_wflow_units(ds_merge, conv_params, dt)

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

#%%We save the output
ds_merge.to_netcdf(outfile_path, encoding=encoding)
ds_merge.close()
print("Done")

