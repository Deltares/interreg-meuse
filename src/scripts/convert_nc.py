import xarray as xr
import os, glob, sys
#import dask

# We read the snakemake parameters
file_temp = snakemake.input.fn_temp
file_pet = snakemake.input.fn_pet
file_precip = snakemake.input.fn_precip
conv_params = snakemake.params.conv_params
year_name = snakemake.params.year_name
dt = snakemake.params.dt_step
outfile_path = str(snakemake.output)

print("------- Checking what we got ------")
print("Outfile path", outfile_path)
print("file_temp", file_temp)
print("file_pet", file_pet)
print("file_precip", file_precip)
print("conv_param", conv_params)
print("dt", dt)
print("year_name", year_name)

#%%
#We open the datasets
ds_merge = xr.open_mfdataset([file_temp, file_pet, file_precip], chunks='auto')

if 'height' in ds_merge.coords.keys():
    ds_merge = ds_merge.squeeze().drop_vars('height') #We remove the height dimension - not needed here

#We rename the variables in accordance to wflow standards
for var in ds_merge.data_vars:
    print(var)
    ds_merge = ds_merge.rename({var: conv_params[var]['wflow_name']})

# TODO - unit conversions - CHECK THIS!
ds_merge['precip'] = ds_merge['precip'] * conv_params['precip'][dt] #going from kg/m2/s to mm/tijdstap
ds_merge['pet'] = ds_merge['pet'] * conv_params['pet'][dt] #going from kg/m2/s to mm/tijdstap
ds_merge['temp'] = ds_merge['temp'] + conv_params['t2m'][dt] #going from Kelvin to C

#encoding otherwise wflow doesn't run
chunksizes = (1, ds_merge.lat.size, ds_merge.lon.size)
encoding = {
        v: {"zlib": True, "dtype": "float32", "chunksizes": chunksizes}
        for v in ds_merge.data_vars.keys()
    }
encoding["time"] = {"_FillValue": None}

#%%We save the output
#fn_out = os.path.join(f"{f_wflow_input}/{dt}/{member_nb}/ds_merged_{year_name}.nc")
ds_merge.to_netcdf(outfile_path, encoding=encoding)

#ADDING SOME PLOTTING!!!

ds_merge.close()
print("Done")

