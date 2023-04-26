import xarray as xr
import os
import matplotlib.pyplot as plt
import numpy as np

# updates the following parameters
#  KsatHorFrac, RiverDepth, wflow_riverwidth, KsatVer, MaxLeakage,

folder_meuse = r'p:\11208719-interreg\wflow\g_rur'
folder_rur = r'p:\11208719-interreg\data\catchment_rur'

file_meuse = 'staticmaps.nc'
file_rur = 'model_d.nc'

ds_meuse = xr.open_dataset(os.path.join(folder_meuse, file_meuse))
ds_rur = xr.open_dataset(os.path.join(folder_rur, file_rur))

# preprocessing needed
# get first and last latitude values from the geul
lat0 = ds_rur.lat[0].squeeze()
latn = ds_rur.lat[-1].squeeze()

# get corresponding latitude values in the meuse
lat_0 = ds_meuse.lat.sel(lat=lat0, method='nearest').squeeze()
lat_n = ds_meuse.lat.sel(lat=latn, method='nearest').squeeze()
lat_sel = ds_meuse.lat.sel(lat=slice(lat_0, lat_n)).values

# replace the original latitude values from the rur with the corresponding ones from the meuse
ds_rur['lat'] = lat_sel

# repeat for lon
# get first and last latitude values from the rur
lon0 = ds_rur.lon[0].squeeze()
lonn = ds_rur.lon[-1].squeeze()

# get corresponding longitude values in the meuse
lon_0 = ds_meuse['lon'].sel(lon=lon0, method='nearest').squeeze()
lon_n = ds_meuse['lon'].sel(lon=lonn, method='nearest').squeeze()
lon_sel = ds_meuse['lon'].sel(lon=slice(lon_0, lon_n)).values

# replace the original longitude values from the rur with the corresponding ones from the meuse
ds_rur['lon'] = lon_sel

# create a copy of the original wflow meuse staticmaps which will be modified
ds_meuse_mod = ds_meuse.copy(deep=True)


def update_var(old_var, new_var):
    ds_rur[new_var] = ds_rur[new_var].fillna(-999)
    ds_meuse_mod[old_var].loc[dict(lat=lat_sel, lon=lon_sel)] = ds_rur[new_var]
    mask = (ds_meuse_mod[old_var] == -999)
    ds_meuse_mod[old_var] = xr.where(mask, ds_meuse[old_var], ds_meuse_mod[old_var])
    return ds_meuse_mod[old_var]


# plt.figure()
# ds_meuse['wflow_uparea'].plot()
# plt.figure()
# ds_rur['wflow_uparea'].plot()
# max wflow uparea is 2345
# base model uses upstream area of >= 25 km2 whereas rur model uses >= 5 km2
# ds_rur['RiverDepth'].where(ds_rur['wflow_uparea'] >= 25).plot()

plt.figure()
ds_meuse["RiverDepth_sobek_global_extrapol"].plot()
# RiverDepth
plt.figure()
ds_rur['RiverDepth'].plot()
ds_rur['RiverDepth'] = ds_rur['RiverDepth'].where(ds_rur['wflow_uparea'] >= 25)
update_var("RiverDepth_sobek_global_extrapol", "RiverDepth")
plt.figure()
ds_meuse_mod['RiverDepth_sobek_global_extrapol'].plot()

# wflow_riverwidth
plt.figure()
ds_meuse["wflow_riverwidth_sobek_global_extrapol"].plot()
ds_rur['wflow_riverwidth'].plot()
ds_rur['wflow_riverwidth'] = ds_rur['wflow_riverwidth'].where(ds_rur['wflow_uparea'] >= 25)
update_var("wflow_riverwidth_sobek_global_extrapol", "wflow_riverwidth")
plt.figure()
ds_meuse_mod["wflow_riverwidth_sobek_global_extrapol"].plot()

plt.figure()
ds_meuse["MaxLeakage_manual_cal"].plot()
# MaxLeakage seems good!
plt.figure()
update_var("MaxLeakage_manual_cal", "MaxLeakage")
ds_meuse_mod["MaxLeakage_manual_cal"].plot()

plt.figure()
ds_meuse["KsatVer"].plot()
# KsatVer also seems good!
plt.figure()
update_var("KsatVer", "KsatVer")
ds_meuse_mod["KsatVer"].plot()

plt.figure()
ds_meuse["ksathorfrac_sub"].plot()
# KsatVer also seems good!
plt.figure()
update_var("ksathorfrac_sub", "KsatHorFrac")
ds_meuse_mod["ksathorfrac_sub"].plot()

plt.close()

folder_save = r'p:\11208719-interreg\wflow\g_rur'
ds_meuse_mod.to_netcdf(os.path.join(folder_save, 'staticmaps.nc'))
