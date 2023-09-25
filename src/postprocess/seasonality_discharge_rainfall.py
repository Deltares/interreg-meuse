#%% To run with hydromt-wflow
import pandas as pd
import matplotlib.pyplot as plt
import pyextremes as pyex
from datetime import datetime
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, glob
from scipy.stats import gumbel_r, genextreme
import xarray as xr
import geopandas as gpd
from hydromt_wflow import WflowModel
import matplotlib.pyplot as plt
from datetime import date, timedelta   
import hydromt  

#%% We import the modelled data
Folder_start = "/p/11208719-interreg"
model_wflow = "p_geulrur"
Folder_p = os.path.join(Folder_start, "wflow", model_wflow)
folder = "members_bias_corrected_daily"
dt = folder.split("_")[-1]
fn_fig = os.path.join(Folder_start, "Figures", model_wflow, folder)
#%%
#We import the Wflow model
fn_config = os.path.join(Folder_p, folder, 'r10i1p5f1', 'members_bias_corrected_daily_r10i1p5f1.toml')
mod = WflowModel(root=os.path.join(Folder_p), mode="r+", config_fn=fn_config)
#mod.staticgeoms.keys()
#mod.staticmaps

if not os.path.exists(fn_fig):
    os.makedirs(fn_fig)

#%%Important locations
dict_cases = {'Rur at Monschau': {'lat': 50.55000 , 'lon': 6.25250, 'id': 15300002, 'source': 'hygon'},
                 'Geul at Meerssen': {'lat': 50.89167, 'lon': 5.72750, 'id': 1036, 'source': 'wl'},
                 'Meuse at Goncourt': {'lat': 48.24167, 'lon': 5.61083, 'id': 1022001001, 'source': 'hp'},
                 'Vesdre at Chaudfontaine': {'lat': 50.59167, 'lon': 5.65250, 'id': 6228, 'source': 'spw'},
                 'Ourthe at Tabreux': {'lat': 50.44167, 'lon': 5.53583, 'id': 5921, 'source': 'spw'},
                 'Sambre at Salzinne': {'lat': 50.45833, 'lon': 4.83583, 'id': 7319, 'source': 'spw'},
                 'Meuse at Chooz': {'lat': 50.09167, 'lon': 4.78583, 'id':1720000001, 'source': 'hp'},
                 'Meuse at St Pieter': {'lat': 50.85000 , 'lon': 5.69417, 'id': 16, 'source': 'rwsinfo'},
                 'Geul at Hommerich': {'lat': 50.80000, 'lon': 5.91917, 'id': 1030, 'source': 'wl'},
                 'Viroin at Treignes': {'lat': 50.09167, 'lon': 4.67750, 'id': 9021, 'source': 'spw'},
                 'Ambleve at Martinrive': {'lat': 50.48333, 'lon': 5.63583, 'id': 6621, 'source': 'spw'},
                 'Semois at Membre Pont': {'lat': 49.86667, 'lon': 4.90250, 'id': 9434, 'source': 'spw'},
                 'Lesse at Gendron': {'lat': 50.20833, 'lon': 4.96083, 'id': 8221, 'source': 'spw'},
                 'Rur at Stah': {'lat': 51.1, 'lon': 6.10250, 'id': 91000001, 'source': 'hygon'},
                 'Meuse at St-Mihiel': {'lat': 48.86667, 'lon': 5.52750, 'id': 12220010, 'source': 'france'}}
#%% We load the shapefiles
#%%
shp_catch = {
        "Rur at Monschau": {"subcatch_hygon":15300002}, #SPW,  
        'Geul at Meerssen': {"subcatch_waterschaplimburg_1036":1036},
        'Meuse at Goncourt': {'subcatch_hydroportail':1022001001},
        'Vesdre at Chaudfontaine': {"subcatch_spw_6228":6228},# SPW, "Ambleve at Martinrive"
        'Ourthe at Tabreux': {"subcatch_spw_5921":5921},# SPW, "Ourthe at Tabreux"    
        'Sambre at Salzinne': {"subcatch_spw_7319":7319},# 
        'Geul at Hommerich': {"subcatch_waterschaplimburg_1030": 1030},     
        'Ambleve at Martinrive': {"subcatch_S01":11},
        'Viroin at Treignes': {'subcatch_spw_9021':9021}, 
        'Semois at Membre Pont': {"subcatch_spw_9434":9434},
        'Lesse at Gendron': {"subcatch_S02":801},
        "Rur at Stah": {"subcatch_hygon_91000001":91000001},       
        'Meuse at St-Mihiel': {"subcatch_S02":101}, 
        'Meuse at Chooz': {"subcatch_hydroportail_1720000001":1720000001}, #hydrofrance et SPW, "Meuse at Chooz"
        'Meuse at St Pieter': {"subcatch_rwsinfo": 16}, #hydrofrance, "Meuse at Goncourt"
} 

#%% We store and extract the rainfall nc files for those stations
fn_in = r'/p/11208719-interreg/data/racmo/members_bias_corrected_revised/c_wflow/daily'
members = [f'r{int(i)}i1p5f1' for i in np.arange(1,17,1)]

fn_out_main = r'/p/11208719-interreg/data/racmo/members_bias_corrected_revised/d_rainfall_analysis'
# %%
# for label in shp_catch.keys(): #['Geul at Hommerich']:#shp_catch.keys():
#         print(label)
#         key = list(shp_catch[label].keys())
#         print(key)
#         df = mod.staticgeoms[key[0]]
#         df.rename(columns={df.columns[0]:'value'}, inplace = True)
#         max = gpd.GeoDataFrame(geometry=df.set_index('value').loc[shp_catch[label][key[0]]], crs= mod.staticgeoms[key[0]].crs) 

#         id = dict_cases[label]['id']
#         fn_out = os.path.join(fn_out_main, str(int(id)))
#         if not os.path.exists(fn_out):
#                 os.makedirs(fn_out)

#         mask_shp = mod.staticmaps['wflow_dem'].raster.geometry_mask(max)
#         all_members = list()
#         for member in members:
#             print(member)
#             all_rainfalls = list()
#             for year in np.arange(1950,2015,1):
#                 #print(year)
#                 ds = xr.open_dataset(os.path.join(fn_in, member, f'ds_merged_{year}.nc'))
#                 mask_shp['lon'] = ds['lon']
#                 mask_shp['lat'] = ds['lat']
#                 masked_precip = xr.where(mask_shp, ds['precip'], np.nan)
#                 clipped_precip = masked_precip.raster.clip_geom(max)
#                 df_precip = clipped_precip.mean(dim=["lat","lon"], skipna=True)
#                 all_rainfalls.append(df_precip)

#             # Concatenate the datasets along the time dimension
#             ds_member_precip = xr.concat(all_rainfalls, dim="time")
#             ds_member_precip = ds_member_precip.to_dataset(name='precip').assign_coords({"stations":id}).expand_dims("stations")
#             ds_member_precip = ds_member_precip.assign_coords({"runs":member}).expand_dims("runs")
#             all_members.append(ds_member_precip)

#         ds_final = xr.concat(all_members, dim="runs")
#         #encoding otherwise wflow doesn't run
#         # chunksizes = (1, ds_member_precip.lat.size, ds_member_precip.lon.size)
#         # encoding = {v: {"zlib": True, "dtype": "float32", "chunksizes": chunksizes} for v in ds_member_precip.data_vars.keys()}
#         # encoding["time"] = {"_FillValue": None}
#         ds_final.to_netcdf(os.path.join(fn_out,f'ds_precip.nc')) #, encoding=encoding)
#         print("Done!")

#%% We perform the statistics
fn_in = r'/p/11208719-interreg/data/racmo/members_bias_corrected_revised/d_rainfall_analysis'
for label in shp_catch.keys(): #['Geul at Hommerich']:#shp_catch.keys():
        print(label)