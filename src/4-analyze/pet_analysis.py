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
case_penman = 'run_penman'
staticmaps = r'eobs_v25.0e_1980_2020.nc'

ds_debruin = xr.open_dataset(os.path.join(cdir, case_debruin, staticmaps))
ds_makkink = xr.open_dataset(os.path.join(cdir, case_makkink, staticmaps))
ds_penman = xr.open_mfdataset(paths=os.path.join(cdir, case_penman, '*.nc'), combine='nested', concat_dim='time', engine='netcdf4')

method = 'makkink'
ds = ds_makkink

# mean monthly values
pet_monthly = ds['pet'].resample(time='m').sum('time').groupby('time.month').mean('time')
# when summing, np.nan -> 0, to exclude
pet_monthly = pet_monthly.where(pet_monthly != 0)
vmin = 10
vmax = 100

fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(11, 6))
month_name = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
              'November', 'December']
for i in range(12):
    month = i + 1  # starts at month 1
    # to remove tick labels, keep y only for column 0 and keep x only for row 0
    if i % 6 > 0:
        axes[i // 6, i % 6].yaxis.set_ticklabels([])

    if i//6 < 1:
        axes[i // 6, i % 6].xaxis.set_ticklabels([])

    if i % 6 > 4:
        pet_monthly.sel(month=month).plot.pcolormesh(
            ax=axes[i // 6, i % 6], cmap='Spectral_r', add_colorbar=True, vmin=vmin, vmax=vmax, extend='both')
    else:
        pet_monthly.sel(month=month).plot.pcolormesh(
            ax=axes[i // 6, i % 6], cmap='Spectral_r', add_colorbar=False, vmin=vmin, vmax=vmax, extend='both')

    axes[i // 6, i % 6].set_title(month_name[i])
    axes[i // 6, i % 6].set_xlabel('')
    axes[i // 6, i % 6].set_ylabel('')

plt.tight_layout()
plt.savefig(cdir + '/figures/monthly_pet_' + method + '.png')

# mean annual values
start_year = 1980
end_year = 2021
pet_yearly = ds['pet'].resample(time='y').sum('time').groupby('time.year').mean('time')
pet_yearly = pet_yearly.where(pet_yearly != 0)
vmin = 520
vmax = 720

fig, axes = plt.subplots(nrows=4, ncols=11, figsize=(12, 8.5))
for i in range(end_year + 1 - start_year):
    year = np.arange(start_year, end_year + 1, 1)

    if i % 11 > 0:
        axes[i//11, i % 11].yaxis.set_ticklabels([])
    if i // 11 < 3:
        axes[i//11, i % 11].xaxis.set_ticklabels([])

    if i % 11 > 9:
        pet_yearly.sel(year=year[i]).plot.pcolormesh(
                ax=axes[i//11, i % 11], cmap='Spectral_r', add_colorbar=True, vmin=vmin, vmax=vmax, extend='both')
    else:
        pet_yearly.sel(year=year[i]).plot.pcolormesh(
                ax=axes[i//11, i % 11], cmap='Spectral_r', add_colorbar=False, vmin=vmin, vmax=vmax, extend='both')

    axes[i//11, i % 11].set_title(str(year[i]))
    axes[i//11, i % 11].set_xlabel('')
    axes[i//11, i % 11].set_ylabel('')

axes[3, 9].set_visible(False)
axes[3, 10].set_visible(False)

plt.tight_layout()
plt.savefig(cdir + '/figures/yearly_pet_' + method + '.png')

# daily pattern for one year
fig, ax = plt.subplots()
pet_daily = ds['pet'].sel(time=slice("1981-01-01", "1981-12-31")).mean(("lon", "lat"))
pet_daily.plot()
ax.set_ylabel('pet [mm]')
ax.set_ylim([0, 4.7])

plt.tight_layout()
plt.savefig(cdir + '/figures/1981_pet_' + method + '.png')
