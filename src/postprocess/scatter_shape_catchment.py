#%%
import pandas as pd
import matplotlib.pyplot as plt
import pyextremes as pyex
from datetime import datetime
import numpy as np
from matplotlib.pyplot import cm
from matplotlib import colors
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, glob
from scipy.stats import gumbel_r, genextreme
import xarray as xr
import math
from matplotlib.pyplot import cm
import hydromt
from hydromt_wflow import WflowModel
import geopandas as gpd
import pickle
import sys
#%% PLotting pop up
%matplotlib qt

def plot_basin(axs, mod, aspect):
    for ax in axs.reshape(-1):
        mod.basins.boundary.plot(ax = ax, color='k', lw=0.1, aspect=aspect, facecolor='gainsboro', zorder = -1) #, aspect = None


#%%
# Folder_start = r"/p/11208719-interreg"
# model_wflow = "p_geulrur" #"o_rwsinfo"
# Folder_p = os.path.join(Folder_start, "wflow", model_wflow)
# folder = "members_bias_corrected_revised_daily"
# fn_fig = os.path.join(Folder_start, "Figures", model_wflow, folder)
# fn_data_out = os.path.join(fn_fig, 'data')

# if not os.path.exists(fn_fig):
#     os.makedirs(fn_fig)
  
# if not os.path.exists(fn_data_out):
#     os.makedirs(fn_data_out)

#%% read one model 
# root = r"/p/11208719-interreg/wflow/p_geulrur"
# config_fn = "members_bias_corrected_revised_daily/r11i1p5f1/members_bias_corrected_revised_daily_r11i1p5f1.toml"
# yml = r"/p/11208719-interreg/data/data_meuse_linux.yml"
# mod = WflowModel(root = root, config_fn=config_fn, data_libs=["deltares_data", yml], mode = "r")

#%%
# We load the shape parameter map
fn = r'c:\Users\couasnon\OneDrive - Stichting Deltares\Documents\PROJECTS\INTERREG\data\Figures\members_bias_corrected_revised_daily\data\GEV_year_params_1040years.nc'
ds = xr.open_dataset(fn)
#ds = xr.open_dataset(os.path.join(fn_data_out,'GEV_year_params_1040years.nc')).load()
ds['shape'] = ds['Q'].sel(dparams='c')

#%% We load the staticmaps
fn = r'c:\Users\couasnon\OneDrive - Stichting Deltares\Documents\PROJECTS\INTERREG\data\wflow\p_geulrur\staticmaps.nc'
#fn = r'/p/11208719-interreg/wflow/p_geulrur/staticmaps.nc'
ds_s = xr.open_dataset(fn).load()
#ds_s = mod.staticmaps.copy()

# %%
ds['uparea'] = ds_s['wflow_uparea']
#variables = Slope? RiverSlope? wflow_uparea

mask = ds.sel(dparams='c')['Q'].isnull()
ds['uparea'] = ds['uparea'].where(~mask)

ds2 = ds.drop('Q').drop_dims('dparams')
ds2['RiverSlope'] = ds_s['RiverSlope']
ds2['log_riverslope'] = np.log(ds_s["RiverSlope"])
ds2['log_uparea'] = np.log(ds["uparea"])
ds2['wflow_streamorder'] = ds_s["wflow_streamorder"]
ds2['slope'] = ds_s['Slope']
ds2['wflow_riverlength'] = ds_s['wflow_riverlength']
ds2['wflow_dem'] = ds_s['wflow_dem'].where(~mask)
ds2['dem_subgrid'] = ds_s['dem_subgrid'].where(~mask)
ds2['RiverDepth'] = ds_s['RiverDepth']
ds2['hydrodem_avg'] = ds_s['hydrodem_avg'].where(~mask)
#%%
levels = np.arange(-0.4,0.5,0.1)
fig, axs = plt.subplots(ncols=3, nrows=1, sharex=True, sharey=True) # figsize=(8,10), 
ds2['shape'].plot(ax=axs[0], levels = levels, center = 0, cbar_kwargs={"label": "shape parameter", "ticks": levels})
ds_s['RiverSlope'].plot(ax=axs[1])
#ds2['log_riverslope'].plot(ax=axs[1])
ds_s['wflow_dem'].plot(ax=axs[2])
plot_basin(axs, mod, aspect=None)
plt.show()

# %%
plt.figure()
ds2.plot.scatter(x='log_riverslope', y='log_uparea', z='wflow_dem', hue='shape', 
                 edgecolors='lightgrey', linewidths=0.5, vmin= -0.2, vmax=0.2, cmap = 'RdBu_r', alpha=0.5) #vmin= -0.2, vmax=0.2, cmap = 'RdBu_r', 

# %%
plt.figure()
ds2.plot.scatter(x='RiverSlope', y='uparea', markersize='RiverDepth', yscale='log', xscale='log', hue='shape',
                 vmin= -0.2, vmax=0.2, cmap = 'RdBu_r')

# %%
plt.figure()
ds2.plot.scatter(x='uparea', y='shape', z='wflow_dem', xscale='log',  hue='log_riverslope')
plt.ylim(-0.5,0.5)
# %%

plt.figure()
ds2.plot.scatter(x='uparea', y='shape', xscale='log',  hue='wflow_streamorder', edgecolors=None) #, {'mec':'k'})

# %%
plt.figure()
ds2.plot.scatter(x='uparea', y='wflow_streamorder', xscale='log',  hue='shape', vmin=-0.2, vmax = 0.2, edgecolors=None) #, {'mec':'k'})

# %%
plt.figure()
ds2.plot.scatter(x='uparea', y='RiverSlope', xscale='log',  yscale = 'log',hue='shape', 
                 vmin= -0.2, vmax=0.2, cmap = 'RdBu_r', edgecolors=None) #, {'mec':'k'})

# %%
