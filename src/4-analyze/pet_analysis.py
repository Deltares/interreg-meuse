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
ds_penman = xr.open_mfdataset(paths=os.path.join(cdir, case_penman, '*.nc'), combine='nested', concat_dim='time', engine='netcdf4')

color_debruin = 'r'
color_makkink = 'g'
color_penman = 'b'

method = 'makkink'
ds = ds_makkink

# mean monthly values


def mean_monthly_sum(ds):
    pet_monthly = ds['pet'].resample(time='m').sum('time').groupby('time.month').mean('time')
    pet_monthly = pet_monthly.where(pet_monthly != 0).mean(("lon", "lat"))
    return pet_monthly


# # monthly sum for all years
fig, ax = plt.subplots()

ax.plot(mean_monthly_sum(ds_debruin), label='debruin', color=color_debruin)
ax.plot(mean_monthly_sum(ds_makkink), label='makkink', color=color_makkink)
ax.plot(mean_monthly_sum(ds_penman), label='penman', color=color_penman)

# converts number to month (abbreviation)
# .strftime("%b")
# ax.xaxis.set_major_locator(mdates.MonthLocator())
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
locs, labels = plt.xticks()
locs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.xticks(locs, labels)
#
# print(mean_monthly_sum(ds_debruin))
#
# ax.set_ylabel('pet [mm/month]')
# plt.legend()
# plt.tight_layout()

# vmin = 10
# vmax = 100
#
# fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(11, 6))
# month_name = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
#               'November', 'December']
# for i in range(12):
#     month = i + 1  # starts at month 1
#     # to remove tick labels, keep y only for column 0 and keep x only for row 0
#     if i % 6 > 0:
#         axes[i // 6, i % 6].yaxis.set_ticklabels([])
#
#     if i//6 < 1:
#         axes[i // 6, i % 6].xaxis.set_ticklabels([])
#
#     if i % 6 > 4:
#         pet_monthly.sel(month=month).plot.pcolormesh(
#             ax=axes[i // 6, i % 6], cmap='Spectral_r', add_colorbar=True, vmin=vmin, vmax=vmax, extend='both')
#     else:
#         pet_monthly.sel(month=month).plot.pcolormesh(
#             ax=axes[i // 6, i % 6], cmap='Spectral_r', add_colorbar=False, vmin=vmin, vmax=vmax, extend='both')
#
#     axes[i // 6, i % 6].set_title(month_name[i])
#     axes[i // 6, i % 6].set_xlabel('')
#     axes[i // 6, i % 6].set_ylabel('')
#
# plt.tight_layout()
# plt.savefig(cdir + '/figures/monthly_pet_' + method + '.png')

# mean seasonal values
# seasons: DJF, JJA, MAM, SON,


def mean_seasonal_sum(ds):
    # pet_seasonal = ds['pet'].resample(time='3m', loffset='-1m').sum('time').groupby('time.season').mean('time')
    pet_seasonal = ds['pet'].resample(time='QS-DEC').sum('time').groupby('time.season').mean('time')
    # 169 groups with labels 1979-12-31, ..., 2021-12-31 is this good? or should I do -2m?
    # bin edge to label bucket with, the default is left
    # resample(time='QS-DEC')
    pet_seasonal = pet_seasonal.where(pet_seasonal != 0)
    return pet_seasonal


def change(ref, obs):
    return (obs - ref) / ref * 100


pet_seasonal_makkink = mean_seasonal_sum(ds_makkink)
pet_seasonal_debruin = change(pet_seasonal_makkink, mean_seasonal_sum(ds_debruin))
pet_seasonal_penman = change(pet_seasonal_makkink, mean_seasonal_sum(ds_penman))

vmin = 100
vmax = 300

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(8, 8))
for i, season in enumerate(("DJF", "MAM", "JJA", "SON")):
    if i > 2:
        pet_seasonal_makkink.sel(season=season).plot.pcolormesh(
            ax=axes[0, i], cmap='Spectral_r', add_colorbar=True, vmin=vmin, vmax=vmax,
            cbar_kwargs={'label': 'makkink pet [mm/season]'})
        pet_seasonal_debruin.sel(season=season).plot.pcolormesh(
            ax=axes[1, i], cmap='Spectral_r', add_colorbar=True, vmin=0, vmax=50,
            cbar_kwargs={'label': 'change debruin [%]'})
        pet_seasonal_penman.sel(season=season).plot.pcolormesh(
            ax=axes[2, i], cmap='Spectral_r', add_colorbar=True, vmin=-20, vmax=20,
            cbar_kwargs={'label': 'change penman [%]'})
    else:
        pet_seasonal_makkink.sel(season=season).plot.pcolormesh(
            ax=axes[0, i], cmap='Spectral_r', add_colorbar=False, vmin=vmin, vmax=vmax)
        pet_seasonal_debruin.sel(season=season).plot.pcolormesh(
            ax=axes[1, i], cmap='Spectral_r', add_colorbar=False, vmin=0, vmax=50)
        pet_seasonal_penman.sel(season=season).plot.pcolormesh(
            ax=axes[2, i], cmap='Spectral_r', add_colorbar=False, vmin=-20, vmax=20)
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

# vmin = 520
# vmax = 720

# fig, axes = plt.subplots(nrows=4, ncols=11, figsize=(12, 8.5))
# for i in range(end_year + 1 - start_year):
#     year = np.arange(start_year, end_year + 1, 1)
#
#     if i % 11 > 0:
#         axes[i//11, i % 11].yaxis.set_ticklabels([])
#     if i // 11 < 3:
#         axes[i//11, i % 11].xaxis.set_ticklabels([])
#
#     if i % 11 > 9:
#         pet_yearly.sel(year=year[i]).plot.pcolormesh(
#                 ax=axes[i//11, i % 11], cmap='Spectral_r', add_colorbar=True, vmin=vmin, vmax=vmax, extend='both')
#     else:
#         pet_yearly.sel(year=year[i]).plot.pcolormesh(
#                 ax=axes[i//11, i % 11], cmap='Spectral_r', add_colorbar=False, vmin=vmin, vmax=vmax, extend='both')
#
#     axes[i//11, i % 11].set_title(str(year[i]))
#     axes[i//11, i % 11].set_xlabel('')
#     axes[i//11, i % 11].set_ylabel('')
#
# axes[3, 9].set_visible(False)
# axes[3, 10].set_visible(False)
#
# plt.tight_layout()
# plt.savefig(cdir + '/figures/yearly_pet_' + method + '.png')
#
# # daily pattern for one year
# fig, ax = plt.subplots()
# pet_daily = ds['pet'].sel(time=slice("1981-01-01", "1981-12-31")).mean(("lon", "lat"))
# pet_daily.plot()
# ax.set_ylabel('pet [mm]')
# ax.set_ylim([0, 4.7])
#
# plt.tight_layout()
# plt.savefig(cdir + '/figures/1981_pet_' + method + '.png')

plt.show()