
import os
import xarray as xr
import numpy as np
import rioxarray as rio
import matplotlib.pyplot as plt

## read staticmaps files
folder_meuse = r'p:\11208719-interreg\wflow\b_rootzone'
folder_geul = r'p:\11208719-interreg\wflow\wflow_sbm_geul_calib'
file = 'staticmaps.nc'

ds_meuse = xr.open_dataset(os.path.join(folder_meuse, file))
ds_geul = xr.open_dataset(os.path.join(folder_geul, file))

## preprocess ds_geul

# rename x and y coordinates to match ds_meuse
ds_geul = ds_geul.rename({'x': 'lon'})
ds_geul = ds_geul.rename({'y': 'lat'})

# get first and last latitude values from the geul
lat0 = ds_geul.lat[0].squeeze()
latn = ds_geul.lat[-1].squeeze()

# get corresponding latitude values in the meuse
lat_0 = ds_meuse.lat.sel(lat=lat0, method='nearest').squeeze()
lat_n = ds_meuse.lat.sel(lat=latn, method='nearest').squeeze()
lat_sel = ds_meuse.lat.sel(lat=slice(lat_0, lat_n)).values

# replace the original latitude values from the geul with the corresponding ones from the meuse
ds_geul['lat'] = lat_sel

# repeat for lon
# get first and last latitude values from the geul
lon0 = ds_geul.lon[0].squeeze()
lonn = ds_geul.lon[-1].squeeze()

# get corresponding latitude values in the meuse
lon_0 = ds_meuse['lon'].sel(lon=lon0, method='nearest').squeeze()
lon_n = ds_meuse['lon'].sel(lon=lonn, method='nearest').squeeze()
lon_sel = ds_meuse['lon'].sel(lon=slice(lon_0, lon_n)).values

# replace the original latitude values from the geul with the corresponding ones from the meuse
ds_geul['lon'] = lon_sel

# create a copy of the original wflow meuse staticmaps which will be modified
ds_meuse_mod = ds_meuse.copy(deep=True)

##
# replace ksathorfrac_sub_ardennes and ksathorfrac_sub variables in meuse with KSHF from geul
ds_geul['KSHF'] = ds_geul['KSHF'].fillna(-999)
ds_meuse_mod["ksathorfrac_sub_ardennes"].loc[dict(lat=lat_sel, lon=lon_sel)] = ds_geul['KSHF'] * 2
mask = (ds_meuse_mod["ksathorfrac_sub_ardennes"] == -999 * 2)
ds_meuse_mod["ksathorfrac_sub_ardennes"] = xr.where(mask, ds_meuse['ksathorfrac_sub_ardennes'], ds_meuse_mod["ksathorfrac_sub_ardennes"])
ds_meuse_mod['ksathorfrac_sub'] = ds_meuse_mod["ksathorfrac_sub_ardennes"]
# https://www.earthdatascience.org/courses/use-data-open-source-python/multispectral-remote-sensing/landsat-in-Python/replace-raster-cell-values-in-remote-sensing-images-in-python/


##
# replace MaxLeakage variable in meuse with MaxLeakage_WB_based2 from geul
ds_geul['MaxLeakage_WB_based2'] = ds_geul['MaxLeakage_WB_based2'].fillna(-999)
ds_meuse_mod["MaxLeakage"].loc[dict(lat=lat_sel, lon=lon_sel)] = ds_geul['MaxLeakage_WB_based2']
mask = (ds_meuse_mod["MaxLeakage"] == -999)
ds_meuse_mod["MaxLeakage"] = xr.where(mask, ds_meuse["MaxLeakage"], ds_meuse_mod["MaxLeakage"])


##
# replace SoilMinThickness and SoilThickness variables in meuse with SoilThickness_15_2_25 from geul
ds_geul['SoilThickness_15_2_25'] = ds_geul['SoilThickness_15_2_25'].fillna(-999)
ds_meuse_mod['SoilMinThickness'].loc[dict(lat=lat_sel, lon=lon_sel)] = ds_geul['SoilThickness_15_2_25'] * 1000
mask = (ds_meuse_mod["SoilMinThickness"] == -999 * 1000)
ds_meuse_mod["SoilMinThickness"] = xr.where(mask, ds_meuse["SoilMinThickness"], ds_meuse_mod["SoilMinThickness"])
ds_meuse_mod['SoilThickness'] = ds_meuse_mod["SoilMinThickness"]

##
folder_save = r'p:\11208719-interreg\wflow\c_geul'
ds_meuse_mod.to_netcdf(os.path.join(folder_save, file))

