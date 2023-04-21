import hydromt
import xarray as xr
import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
from hydromt_wflow import WflowModel
import matplotlib.dates as mdates

cdir = r'c:\Projects\interreg\local\interreg-meuse\wflow'
case_debruin = 'run_debruin'
case_makkink = 'run_makkink'
case_penman = 'run_penman'
staticmaps = r'eobs_v25.0e_1980_2020.nc'

ds_debruin = xr.open_dataset(os.path.join(cdir, case_debruin, staticmaps))
ds_makkink = xr.open_dataset(os.path.join(cdir, case_makkink, staticmaps))
ds_penman = xr.open_mfdataset(paths=os.path.join(cdir, case_penman, '*.nc'), combine='nested', concat_dim='time',
                              engine='netcdf4')

color_debruin = 'r'
color_makkink = 'g'
color_penman = 'b'

method = 'makkink'
ds = ds_makkink


def mean_monthly_sum(ds):
    pet_monthly = ds['pet'].resample(time='m').sum('time').groupby('time.month').mean('time')
    pet_monthly = pet_monthly.where(pet_monthly != 0).mean(("lon", "lat"))
    return pet_monthly


# monthly sum for all years
fig, ax = plt.subplots()

ax.plot(mean_monthly_sum(ds_debruin), label='debruin', color=color_debruin)
ax.plot(mean_monthly_sum(ds_makkink), label='makkink', color=color_makkink)
ax.plot(mean_monthly_sum(ds_penman), label='penman', color=color_penman)

locs, labels = plt.xticks()
locs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.xticks(locs, labels)

ax.set_ylabel('PET [mm/month]')
plt.legend()
plt.tight_layout()


def mean_seasonal_sum(ds):
    pet_seasonal = ds['pet'].resample(time='QS-DEC').sum('time').groupby('time.season').mean('time')
    pet_seasonal = pet_seasonal.where(pet_seasonal != 0)
    return pet_seasonal


def change(ref, obs):
    return (obs - ref) / ref * 100


pet_seasonal_makkink = mean_seasonal_sum(ds_makkink)
seasonal_change_debruin = mean_seasonal_sum(ds_debruin) - pet_seasonal_makkink
seasonal_change_penman = mean_seasonal_sum(ds_penman) - pet_seasonal_makkink

vmin = 100
vmax = 300

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(8, 8))
for i, season in enumerate(("DJF", "MAM", "JJA", "SON")):
    if i > 2:
        pet_seasonal_makkink.sel(season=season).plot.pcolormesh(
            ax=axes[0, i], cmap='Spectral_r', add_colorbar=True, vmin=vmin, vmax=vmax,
            cbar_kwargs={'label': 'makkink pet [mm/season]'})
        seasonal_change_debruin.sel(season=season).plot.pcolormesh(
            ax=axes[1, i], cmap='Spectral_r', add_colorbar=True, vmin=0, vmax=20,
            cbar_kwargs={'label': 'Debruin diff [mm/season]'})
        seasonal_change_penman.sel(season=season).plot.pcolormesh(
            ax=axes[2, i], cmap='Spectral_r', add_colorbar=True, vmin=-30, vmax=30,
            cbar_kwargs={'label': 'Penman diff [mm/season]'})
    else:
        pet_seasonal_makkink.sel(season=season).plot.pcolormesh(
            ax=axes[0, i], cmap='Spectral_r', add_colorbar=False, vmin=vmin, vmax=vmax)
        seasonal_change_debruin.sel(season=season).plot.pcolormesh(
            ax=axes[1, i], cmap='Spectral_r', add_colorbar=False, vmin=0, vmax=20)
        seasonal_change_penman.sel(season=season).plot.pcolormesh(
            ax=axes[2, i], cmap='Spectral_r', add_colorbar=False, vmin=-30, vmax=30)
    if i > 0:
        axes[0, i].yaxis.set_ticklabels([])
        axes[0, i].set_ylabel('')
        axes[1, i].yaxis.set_ticklabels([])
        axes[1, i].set_ylabel('')
        axes[2, i].yaxis.set_ticklabels([])
        axes[2, i].set_ylabel('')

    axes[0, i].xaxis.set_ticklabels([])
    axes[0, i].set_xlabel('')
    axes[1, i].xaxis.set_ticklabels([])
    axes[1, i].set_xlabel('')


# mean annual values
start_year = 1980
end_year = 2021
years = np.arange(start_year, end_year+1, 1)


def mean_yearly_sum(ds):
    pet_yearly = ds['pet'].resample(time='y').sum('time').groupby('time.year').mean('time')
    pet_yearly = pet_yearly.where(pet_yearly != 0).mean(("lon", "lat"))
    return pet_yearly


# yearly sum for all years
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(mean_yearly_sum(ds_debruin), drawstyle='steps', label='debruin', color=color_debruin)
ax.plot(mean_yearly_sum(ds_makkink), drawstyle='steps', label='makkink', color=color_makkink)
ax.plot(mean_yearly_sum(ds_penman), drawstyle='steps', label='penman', color=color_penman)

locs, labels = plt.xticks()
locs = range(len(years))
labels = years
plt.xticks(locs, labels, rotation="vertical")

ax.set_ylabel('pet [mm/year]')
plt.legend()
plt.tight_layout()
plt.show()

