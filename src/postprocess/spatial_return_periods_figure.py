#%%
import pandas as pd
import matplotlib.pyplot as plt
from hydromt.stats import extremes
import os
import xarray as xr
import numpy as np
import xclim
from xclim.ensembles import create_ensemble
import glob
import time
from spatial_return_periods import stacking_ensemble
import scipy

print("in script")
st = time.time()

def clean_subaxis(axs):
    for ax in axs.reshape(-1):
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title('')

def month_calc(x):
    from datetime import date
    func = lambda x: date.fromordinal(int(x)).month
    return xr.apply_ufunc(func, x, vectorize=True)


def _mode(*args, **kwargs):
    vals = scipy.stats.mode(*args, **kwargs)
    # only return the mode (discard the count)
    return vals[0].squeeze()


def mode(obj, dim):
    # note: apply always moves core dimensions to the end
    # usually axis is simply -1 but scipy's mode function doesn't seem to like that
    # this means that this version will only work for DataArray's (not Datasets)
    assert isinstance(obj, xr.DataArray)
    axis = obj.ndim - 1
    return xr.apply_ufunc(_mode, obj,
                          input_core_dims=[[dim]],
                          kwargs={'axis': axis})

#%% We load one file for now 
# We import the modelled data
#Folder_start = r"p:/11208719-interreg"
Folder_start = r"/p/11208719-interreg"
model_wflow = "o_rwsinfo"
Folder_p = os.path.join(Folder_start, "wflow", model_wflow)
folder = "members_bias_corrected_revised_daily"
fn_fig = os.path.join(Folder_start, "Figures", model_wflow, folder)
fn_data_out = os.path.join(fn_fig, 'data')

if not os.path.exists(fn_fig):
    os.makedirs(fn_fig)
  
if not os.path.exists(fn_data_out):
    os.makedirs(fn_data_out)

#%% We load the data
mask_riv = xr.open_dataarray(os.path.join(fn_data_out,'mask_rivers_wflow.nc')).load()

#%% # Figure of shape parameters 

#Importing data
params = xr.open_dataset(os.path.join(fn_data_out,'GEV_year_params_per_ensemble.nc')).load()

#Looking at whether shape changes sign across ensembles
shape_positive = xr.where(params['Q'].sel(dparams='c')>0, 1, 0)
shape_positive_sum = shape_positive.sum(dim='realization') 
shape_positive_sum = shape_positive_sum.where(mask_riv)
print("Statistics per ensemble done")

all_params = xr.open_dataarray(os.path.join(fn_data_out,'GEV_year_params_1040years.nc')).load()
#%%
print("- Starting figure plotting for shape params-")

levels = np.arange(-5,5,0.5)

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10,6), sharex=True, sharey=True)
params['Q'].sel(dparams='c').min(dim='realization').plot(ax=axs[0,0], levels=levels, cbar_kwargs={"label": "shape parameter", "ticks": levels}) #, cmap = cmap, vmin = 10, vmax = 300)
params['Q'].sel(dparams='c').max(dim='realization').plot(ax=axs[0,1], levels=levels, cbar_kwargs={"label": "shape parameter", "ticks": levels})#
shape_positive_sum.plot(ax=axs[1,0], levels=np.arange(0,17,1), cbar_kwargs={"label": "Nb. positive", "ticks": np.arange(0,17,1)})
all_params.sel(dparams='c').plot(ax=axs[1,1], levels=levels, cbar_kwargs={"label": "shape parameter", "ticks": levels})

clean_subaxis(axs)

axs[0,0].set_title('a - minimum across ensembles', fontsize=8)
axs[0,1].set_title('b - maximum across ensembles',fontsize=8)
axs[1,0].set_title('c - nb. of positive shape parameter', fontsize=8)
axs[1,1].set_title('d - shape parameter from 1,040 years',fontsize=8)

axs[0,0].set_ylabel('latitude (degrees)', fontsize=8)
axs[1,0].set_ylabel('latitude (degrees)', fontsize=8)

axs[1,0].set_xlabel('longitude (degrees)', fontsize=8)
axs[1,1].set_xlabel('longitude (degrees)', fontsize=8)

fig.savefig(os.path.join(fn_fig, f'spatial_daily_shape_gev.png'), dpi=400) 
plt.close()
print("Figure shape params range saved")

#%% Figure of T return periods 
print("- Starting figure plotting for return periods -")

#Importing data
all_Temp = xr.open_dataset(os.path.join(fn_data_out,'EMP_return_periods_1040years.nc')).load()
all_Tgumb = xr.open_dataset(os.path.join(fn_data_out,'Gumbel_return_periods_1040years.nc')).load()
all_Tgev = xr.open_dataset(os.path.join(fn_data_out,'GEV_return_periods_1040years.nc')).load()


levels_diff = np.arange(-100,120,20)
levels_Q = [0,10,50,100,500,1000,2000,5000,10000]

#Plotting results - Gumbel
all_ds = all_Tgumb
all_diff = (all_ds - all_Temp)*100/all_Temp 

fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(10,18), sharex=True, sharey=True)
all_ds['Q'].sel(return_period=10).plot(ax=axs[0,0], levels=levels_Q, add_colorbar = False)
all_Temp['Q'].sel(return_period=10).plot(ax=axs[1,0], levels=levels_Q, add_colorbar = False)

all_ds['Q'].sel(return_period=50).plot(ax=axs[0,1], levels=levels_Q, add_colorbar = False)
all_Temp['Q'].sel(return_period=50).plot(ax=axs[1,1], levels=levels_Q, add_colorbar = False)

all_ds['Q'].sel(return_period=100).plot(ax=axs[0,2], levels=levels_Q, cbar_kwargs={"label": "Discharge - m3/s", "ticks": levels_Q})
all_Temp['Q'].sel(return_period=100).plot(ax=axs[1,2], levels=levels_Q, cbar_kwargs={"label": "Discharge - m3/s", "ticks": levels_Q})

all_diff['Q'].sel(return_period=10).plot(ax=axs[2,0], center = 0, levels=levels_diff, add_colorbar = False)
all_diff['Q'].sel(return_period=50).plot(ax=axs[2,1], center = 0, levels=levels_diff, add_colorbar = False)
all_diff['Q'].sel(return_period=100).plot(ax=axs[2,2], center = 0, levels=levels_diff, cbar_kwargs={"label": "Perc. difference", "ticks": levels_diff})

clean_subaxis(axs)

axs[0,0].set_title('T = 10 years', fontsize=10)
axs[0,1].set_title('T = 50 years', fontsize=10)
axs[0,2].set_title('T = 100 years', fontsize=10)
axs[0,0].set_ylabel('Gumbel', fontsize=10)
axs[1,0].set_ylabel('Empirical - 1040 years', fontsize=10)
axs[2,0].set_ylabel('Difference (%)', fontsize=10)

fig.savefig(os.path.join(fn_fig, f'spatial_daily_gumbel_emp_diff.png'), dpi=400)
plt.close()
print("Figure Gumbel saved")

#Plotting results - GEV
all_ds = all_Tgev
all_diff = (all_ds - all_Temp)*100/all_Temp

fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(10,18), sharex=True, sharey=True)
all_ds['Q'].sel(return_period=10).plot(ax=axs[0,0], levels=levels_Q, add_colorbar = False)
all_Temp['Q'].sel(return_period=10).plot(ax=axs[1,0], levels=levels_Q, add_colorbar = False)

all_ds['Q'].sel(return_period=50).plot(ax=axs[0,1], levels=levels_Q, add_colorbar = False)
all_Temp['Q'].sel(return_period=50).plot(ax=axs[1,1], levels=levels_Q, add_colorbar = False)

all_ds['Q'].sel(return_period=100).plot(ax=axs[0,2], levels=levels_Q, cbar_kwargs={"label": "Discharge - m3/s", "ticks": levels_Q})
all_Temp['Q'].sel(return_period=100).plot(ax=axs[1,2], levels=levels_Q, cbar_kwargs={"label": "Discharge - m3/s", "ticks": levels_Q})

all_diff['Q'].sel(return_period=10).plot(ax=axs[2,0], center = 0, levels=levels_diff, add_colorbar = False)
all_diff['Q'].sel(return_period=50).plot(ax=axs[2,1], center = 0, levels=levels_diff, add_colorbar = False)
all_diff['Q'].sel(return_period=100).plot(ax=axs[2,2], center = 0, levels=levels_diff, cbar_kwargs={"label": "Perc. difference", "ticks": levels_diff})

clean_subaxis(axs)

axs[0,0].set_title('T = 10 years', fontsize=10)
axs[0,1].set_title('T = 50 years', fontsize=10)
axs[0,2].set_title('T = 100 years', fontsize=10)
axs[0,0].set_ylabel('Gumbel', fontsize=10)
axs[1,0].set_ylabel('Empirical - 1040 years', fontsize=10)
axs[2,0].set_ylabel('Difference (%)', fontsize=10)

fig.savefig(os.path.join(fn_fig, f'spatial_daily_gev_emp_diff.png'), dpi=400)
plt.close()
print("Figure GEV saved")

#%% FFigure of shape parameters - summer, winter, all
print("- Starting figure plotting for seasonal shape parameters -")

#Importing data
summer_params = xr.open_dataset(os.path.join(fn_data_out,'GEV_summer_params_1040years.nc')).load()
winter_params = xr.open_dataset(os.path.join(fn_data_out,'GEV_winter_params_1040years.nc')).load()

fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(10,12), sharex=True, sharey=True)
g = all_params.sel(dparams='c').plot(ax=axs[0], levels=levels, add_colorbar=False) #, cbar_kwargs={"label": "shape parameter", "ticks": levels})
summer_params['Q'].sel(dparams='c').plot(ax=axs[1], levels=levels, add_colorbar=False)#, cbar_kwargs={"label": "shape parameter", "ticks": levels})
winter_params['Q'].sel(dparams='c').plot(ax=axs[2], levels=levels, add_colorbar=False)#, cbar_kwargs={"label": "shape parameter", "ticks": levels})

clean_subaxis(axs)

axs[1].set_ylabel('latitude (degrees)', fontsize=10)
axs[2].set_xlabel('longitude (degrees)', fontsize=10)
axs[0].set_title('a - All year', fontsize=10)
axs[1].set_title('b - Only summer', fontsize=10)
axs[2].set_title('c - Only winter', fontsize=10)

#cax = plt.axes([0.90, 0.1, 0.01, 0.8])
fig.subplots_adjust(right=0.95) # create space on the right hand side
#cax = plt.axes([0.70, 0.1, 0.01, 0.5]) # add a small custom axis: [left, bottom, width, height]
fig.colorbar(g, ax = axs, label='shape parameter', ticks = levels, shrink=0.75)
#plt.tight_layout()

fig.savefig(os.path.join(fn_fig, f'spatial_daily_gev_shape_summer_winter_all.png'), dpi=400)
#plt.close()s

print("Figure shape param per season - done")
#%% Figure of date summer/winter events and average magnitude 

#Importing the data
sub = xr.open_dataset(os.path.join(fn_data_out,'AM_datesAM_Oct.nc')).load()
all_years = stacking_ensemble(sub)
all_years = all_years.where(mask_riv)

is_summer = np.arange(91, 274, 1)
all_summers = all_years.where(all_years['dayofyear'].isin(is_summer))

levels_years = [1,10,20,30,40,50,60,70,80,90,100]
levels_month = np.arange(1,13,1)

color_map = plt.get_cmap('viridis')
color_map.set_under('red')


ds = all_summers['dayofyear'].count(dim='time')*100/len(all_summers['time'])
fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(10,6), sharex=True, sharey=True)
ds.where(mask_riv).plot(ax=axs[0], vmin=1, cmap = color_map, levels=levels_years, cbar_kwargs={"label": "Percentage", "extend": "min"}) #, "ticks": levels_years
all_summers['Q'].mean(dim='time').where(mask_riv).plot(ax=axs[1])

clean_subaxis(axs)
axs[0].set_ylabel('latitude (degrees)', fontsize=10)
axs[1].set_ylabel('latitude (degrees)', fontsize=10)
axs[1].set_xlabel('longitude (degrees)', fontsize=10)

axs[0].set_title('a - Nb of AM in summer', fontsize=10)
axs[1].set_title('b - Average discharge of a)', fontsize=10)
fig.savefig(os.path.join(fn_fig, f'spatial_daily_emp_summer_stats.png'), dpi=400)

print("Figure summer AM events - done")

#Getting the top 10 events using the quantile function
qs = [(len(all_years['z'])-i)/len(all_years['z']) for i in np.arange(0,10,1)]
all_years['top_10'] = all_years['Q'].quantile(qs, dim='time', method='closest_observation')

datasets = []
for i_top in all_years['quantile'].values:
    print(i_top)
    top_event = all_years['month'].where(all_years['Q'] == all_years['top_10'].sel(quantile=i_top))
    datasets.append(top_event.max('time'))
top_events_month = xr.concat(datasets, dim='top')

#mask_top = top_events.fillna(0)
#mask_top = mask_top != 0 
#top_events_month = month_calc(top_events.fillna(1)).where(mask_top)

top_events_mode = mode(top_events_month, dim='top')

fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(10,6), sharex=True, sharey=True)
top_events_mode.where(mask_riv).plot(ax=axs[0], levels=levels_month, cmap = plt.get_cmap("hsv",13), cbar_kwargs={"label": "doy", "ticks":levels_month})
all_years['top_10'].mean(dim='quantile').where(mask_riv).plot(ax=axs[1], cbar_kwargs={"label": "Average discharge (m3/s)"})

clean_subaxis(axs)
axs[0].set_ylabel('latitude (degrees)', fontsize=10)
axs[1].set_ylabel('latitude (degrees)', fontsize=10)
axs[1].set_xlabel('longitude (degrees)', fontsize=10)

axs[0].set_title('a - Mode of month from top 10 events', fontsize=10)
axs[1].set_title('b - Average discharge of top 10 events', fontsize=10)
fig.savefig(os.path.join(fn_fig, f'spatial_daily_emp_top10_stats.png'), dpi=400)
plt.close()

print("Figure top 10 AM events - done")

#%%
print("- Starting figure plotting for Empirical seasonal return periods -")

#Importing data
# all_Temp = xr.open_dataset(os.path.join(fn_data_out,'EMP_return_periods_1040years.nc')).load()
summer_Temp = xr.open_dataset(os.path.join(fn_data_out,'EMP_summer_return_periods_1040years.nc')).load()
winter_Temp = xr.open_dataset(os.path.join(fn_data_out,'EMP_winter_return_periods_1040years.nc')).load()


# levels_diff = np.arange(-100,120,20)
# levels_Q = [0,10,50,100,500,1000,2000,5000,10000]

#Plotting results - EMP
fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(10,18), sharex=True, sharey=True)
g = summer_Temp['Q'].sel(return_period=10).plot(ax=axs[0,0], levels=levels_Q, add_colorbar = False)
winter_Temp['Q'].sel(return_period=10).plot(ax=axs[1,0], levels=levels_Q, add_colorbar = False)

summer_Temp['Q'].sel(return_period=50).plot(ax=axs[0,1], levels=levels_Q, add_colorbar = False)
winter_Temp['Q'].sel(return_period=50).plot(ax=axs[1,1], levels=levels_Q, add_colorbar = False)

summer_Temp['Q'].sel(return_period=100).plot(ax=axs[0,2], levels=levels_Q, add_colorbar = False
winter_Temp['Q'].sel(return_period=100).plot(ax=axs[1,2], levels=levels_Q, add_colorbar = False)

all_Temp['Q'].sel(return_period=10).plot(ax=axs[2,0], center = 0, levels=levels_Q, add_colorbar = False)
all_Temp['Q'].sel(return_period=50).plot(ax=axs[2,1], center = 0, levels=levels_Q, add_colorbar = False)
all_Temp['Q'].sel(return_period=100).plot(ax=axs[2,2], center = 0, levels=levels_Q, add_colorbar = False)

clean_subaxis(axs)

axs[0,0].set_title('T = 10 years', fontsize=10)
axs[0,1].set_title('T = 50 years', fontsize=10)
axs[0,2].set_title('T = 100 years', fontsize=10)
axs[0,0].set_ylabel('Summer', fontsize=10)
axs[1,0].set_ylabel('Winter', fontsize=10)
axs[2,0].set_ylabel('Hydrological year', fontsize=10)

fig.subplots_adjust(right=0.95) # create space on the right hand side
fig.colorbar(g, ax = axs, label='Discharge (m3/s)', ticks = levels_Q, shrink=0.75)


fig.savefig(os.path.join(fn_fig, f'spatial_daily_emp_return_periods_seasonal.png'), dpi=400)
plt.close()
print("Figure empirical seasonal T saved")

#%%
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')




