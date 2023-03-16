import hydromt
import xarray as xr
import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
from hydromt_wflow import WflowModel

cdir = r'c:\Projects\interreg\local\interreg-meuse\wflow'
case_debruin = 'run_debruin'
case_makkink = 'run_makkink'

staticmaps_debruin = r'inmaps_eobs_v25.0e_era5_daily_zarrd_debruin_86400_1980_2021.nc'
staticmaps_makkink = r'inmaps_eobs_v25.0e_era5_daily_zarrd_makkink_86400_1980_2021.nc'
# staticmaps_penman = ''

ds_debruin = xr.open_dataset(os.path.join(cdir, case_debruin, staticmaps_debruin))
ds_makkink = xr.open_dataset(os.path.join(cdir, case_makkink, staticmaps_makkink))
# ds_penman = xr.open_dataset(os.path.join(cdir, case_makkink, staticmaps_penman))

pet_test = ds_debruin['pet'].sel(time=slice("1981-01-01", "1982-01-01")).mean('time')
fig, ax = plt.subplots()
pet_test.plot.pcolormesh(ax=ax, cmap='Spectral_r', add_colorbar=True, extend='both')

pet_test = ds_debruin['pet'].sel(lat=50.1, lon=5, method='nearest')
fig, ax = plt.subplots()
pet_test.plot()

# mean monthly values
# df_m = df.resample('m').sum()
# df_month = df_m.groupby(df_m.index.month).mean()
pet_monthly = ds_debruin['pet'].groupby('time.month').mean('time')
fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(11, 6))
month_name = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
              'November', 'December']
for i in range(12):
    month = i + 1  # starts at month 1
    pet_monthly.sel(month=month).plot.pcolormesh(
        ax=axes[i//6, i%6], cmap='Spectral_r', add_colorbar=True, extend='both')
    axes[i//6, i%6].set_title(month_name[i])
    axes[i//6, i%6].set_xlabel('')
    axes[i//6, i%6].set_ylabel('')

plt.tight_layout()

# mean annual values
start_year = 1980
end_year = 2021
pet_yearly = ds_debruin['pet'].groupby('time.year').mean('time')

fig, axes = plt.subplots(nrows=4, ncols=10, figsize=(11, 6))
for i in range(40):
    year = np.arange(start_year, end_year, 1)
    pet_yearly.sel(year=year[i]).plot.pcolormesh(
        ax=axes[i//10, i % 10], cmap='Spectral_r', add_colorbar=True, extend='both')
    axes[i//10, i % 10].set_title(str(year[i]))
    axes[i//10, i % 10].set_xlabel('')
    axes[i//10, i % 10].set_ylabel('')

# daily pattern for one year
fig, ax = plt.subplots()
pet_daily = ds_debruin['pet'].sel(time=slice("1981-01-01", "1982-01-01")).mean(("lon", "lat"))
pet_daily.plot()
plt.show()