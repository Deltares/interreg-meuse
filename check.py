#%%
import xarray as xr
import dask
dask.config.set(scheduler='threads') 


file_pet = r'/p/11208719-interreg/data/racmo/members_bias_corrected_revised/b_preprocess/hourly/r4i1p5f1/pet/pet.KNMI-2004.r4i1p5f1_regrid_meuse.nc'
file_temp = r'/p/11208719-interreg/data/racmo/members_bias_corrected_revised/b_preprocess/hourly/r4i1p5f1/t2m/t2m.KNMI-2004.r4i1p5f1_regrid_meuse.nc'
file_precip = r'/p/11208719-interreg/data/racmo/members_bias_corrected_revised/b_preprocess/hourly/r4i1p5f1/precip/precip.KNMI-2004.r4i1p5f1_regrid_meuse.nc'

outfile_path = r'/p/11208719-interreg/data/racmo/members_bias_corrected_revised/c_wflow/hourly/r4i1p5f1/ds_merged_2004.nc'

#%%
path_dict={'pet': file_pet, 
            't2m': file_temp,
            'precip': file_precip}

conv_params = {
    'pet': {
        'daily': 86400,
        'hourly': 3600,
        'conversion': "multiplicative",
        'expected_original_units': ['kg m-2 s-1','kg/m2/s'] , #Attributes of the units in the cdo netcdf file
        'new_units': 'mm/timestep',
        'wflow_name': "pet"},
    'precip':{
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
#%%
dt = 'hourly'

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
%%time
#We open the datasets
ds_merge = xr.open_mfdataset([file_temp, file_pet, file_precip], chunks='auto', combine='nested') #.load() # chunks = {"time":1000} #ds.chunks
#ds_merge = xr.open_mfdataset([file_temp, file_pet, file_precip], chunks='auto')
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

#%% We save the output
ds_merge.to_netcdf(outfile_path, encoding=encoding)
ds_merge.close()
print("Done")




# %%
%%time
#We open the datasets
ds_merge = xr.open_mfdataset([file_temp, file_pet, file_precip], chunks = {"time":2000}, combine='nested') # chunks = {"time":1000} #ds.chunks

if 'height' in ds_merge.coords.keys():
    ds_merge = ds_merge.squeeze().drop_vars('height') #We remove the height dimension - not needed here

#We remove the units attributes from the main attributes as this is confusing
del ds_merge.attrs['units']

#We add each attributes as variables attributes
for var in ds_merge.data_vars:
    print(var)
    ds_var = xr.open_dataset(path_dict[var], chunks = {"time":2000})
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

ds_merge.compute()

#encoding otherwise wflow doesn't run
chunksizes = (1, ds_merge.lat.size, ds_merge.lon.size)
encoding = {
        v: {"zlib": True, "dtype": "float32", "chunksizes": chunksizes}
        for v in ds_merge.data_vars.keys()
    }
encoding["time"] = {"_FillValue": None}

#%% We save the output
ds_merge.to_netcdf(outfile_path, encoding=encoding)
ds_merge.close()
print("Done")
# %%
