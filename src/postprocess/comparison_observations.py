#%%
import pandas as pd
import matplotlib.pyplot as plt
import pyextremes as pyex
from datetime import datetime
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
import matplotlib as mpl
import os, glob
from scipy.stats import gumbel_r, genextreme
from hydromt_wflow import WflowModel
import xarray as xr
import math
import geopandas as gpd
import contextily as cx

#%%
# We import the modelled data
Folder_start = "/p/11208719-interreg"
model_wflow = "p_geulrur" #"o_rwsinfo" #"f_spwgauges"
Folder_p = os.path.join(Folder_start, "wflow", model_wflow)
folder = "members_bias_corrected_revised_daily"
fn_fig = os.path.join(Folder_start, "Figures", model_wflow, folder)

if not os.path.exists(fn_fig):
    os.makedirs(fn_fig)

#We import the Wflow model
fn_config = os.path.join(Folder_p, folder, 'r10i1p5f1', 'members_bias_corrected_revised_daily_r10i1p5f1.toml')
mod = WflowModel(root=os.path.join(Folder_p), mode="r+", config_fn=fn_config)
mod.staticgeoms.keys()
#mod.staticmaps

#%%
shp_catch_main = {
        'Meuse at St Pieter': {"subcatch_rwsinfo": 16}, #hydrofrance, "Meuse at Goncourt"
        'Meuse at Chooz': {"subcatch_hydroportail_1720000001":1720000001}, #hydrofrance et SPW, "Meuse at Chooz"
        'Sambre at Salzinne': {"subcatch_spw_7319":7319},# 
        'Ourthe at Tabreux': {"subcatch_spw_5921":5921},# SPW, "Ourthe at Tabreux"        
        'Vesdre at Chaudfontaine': {"subcatch_spw_6228":6228},# SPW, "Ambleve at Martinrive"
        'Meuse at Goncourt': {'subcatch_hydroportail':1022001001},
        'Geul at Meerssen': {"subcatch_waterschaplimburg_1036":1036},
        "Rur at Monschau": {"subcatch_hygon":15300002}, #SPW,                

        # "Rur at Monschau": {"subcatch_hygon":15300002}, #SPW, "Sambre at Salzinnes"    
        # 'Geul at Meerssen': {"subcatch_waterschaplimburg_1036":1036}, # SPW, "
        # 'Meuse at Goncourt': {'subcatch_hydroportail':1022001001}, #/St Pieter -- station recalculée Discharge at Kanne + discharge at St Pieter , "Meuse at Borgharen"
        # 'Vesdre at Chaudfontaine': {"subcatch_spw_6228":6228},# SPW, "Ambleve at Martinrive"
        # 'Ourthe at Tabreux': {"subcatch_spw_5921":5921},# SPW, "Ourthe at Tabreux"
        # 'Sambre at Salzinne': {"subcatch_spw_7319":7319},# 
        # 'Meuse at Chooz': {"subcatch_hydroportail":1720000001}, #hydrofrance et SPW, "Meuse at Chooz"
        # 'Meuse at St Pieter': {"subcatch_rwsinfo": 16}, #hydrofrance, "Meuse at Goncourt"
        } 

#%%
shp_catch_appendix = {
        'Meuse at St-Mihiel': {"subcatch_S02":101}, 
        "Rur at Stah": {"subcatch_hygon_91000001":91000001}, 
        'Lesse at Gendron': {"subcatch_S02":801},
        'Semois at Membre Pont': {"subcatch_spw_9434":9434},
        'Ambleve at Martinrive': {"subcatch_S01":11},
        'Viroin at Treignes': {'subcatch_spw_9021':9021}, 
        'Geul at Hommerich': {"subcatch_waterschaplimburg_1030": 1030}, 

        # 'Geul at Hommerich': {"subcatch_waterschaplimburg_1030": 1030}, 
        # 'Viroin at Treignes': {'subcatch_spw_9021':9021}, 
        # 'Ambleve at Martinrive': {"subcatch_S01":11},
        # 'Semois at Membre Pont': {"subcatch_spw_9434":9434},
        # 'Lesse at Gendron': {"subcatch_S02":801},
        # "Rur at Stah": {"subcatch_hygon_91000001":91000001}, 
        # 'Meuse at St-Mihiel': {"subcatch_S02":101}, 
        } 

#%%
shp_catch = shp_catch_appendix
#%%
# fn_runs = glob.glob(os.path.join(Folder_p, folder, '*', 'output.csv'))
# date_parser = lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S")

# model_runs = {
#     "r1": {"case":folder, 
#             "folder": "r1i1p5f1"},
#     "r2": {"case":folder, 
#             "folder": "r2i1p5f1"},
#     "r3": {"case":folder, 
#             "folder": "r3i1p5f1"},
#     "r4": {"case":folder, 
#             "folder": "r4i1p5f1"}, 
#     "r5": {"case":folder, 
#             "folder": "r5i1p5f1"}, 
#     "r6": {"case":folder, 
#             "folder": "r6i1p5f1"}, 
#     "r7": {"case":folder, 
#             "folder": "r7i1p5f1"}, 
#     "r8": {"case":folder, 
#             "folder": "r8i1p5f1"}, 
#     "r9": {"case":folder, 
#             "folder": "r9i1p5f1"}, 
#     "r10": {"case":folder, 
#             "folder": "r10i1p5f1"}, 
#     "r11": {"case":folder, 
#             "folder": "r11i1p5f1"}, 
#     "r12": {"case":folder, 
#             "folder": "r12i1p5f1"}, 
#     "r13": {"case":folder, 
#             "folder": "r13i1p5f1"}, 
#     "r14": {"case":folder, 
#             "folder": "r14i1p5f1"}, 
#     "r15": {"case":folder, 
#             "folder": "r15i1p5f1"}, 
#     "r16": {"case":folder, 
#             "folder": "r16i1p5f1"}, 
# } 
# #%%
# shp_catch = {'Q_1011': "subcatch_S01", #hydrofrance, "Meuse at Goncourt"
#         'Q_4': "subcatch_S04", #hydrofrance et SPW, "Meuse at Chooz"
#         'Q_16': 'subcatch_S06', #/St Pieter -- station recalculée Discharge at Kanne + discharge at St Pieter , "Meuse at Borgharen"
#         'Q_12': "subcatch_S01",# SPW - severely hit 2021, "Vesdre at Chaudfontaine"  - SEL ############
#         'Q_11': "subcatch_S01",# SPW, "Ambleve at Martinrive"
#         'Q_10': "subcatch_S02",# SPW, "Ourthe at Tabreux"
#         'Q_801': "subcatch_S02", # SPW, "Lesse at Gendron" - SEL ############
#         "Q_9": "subcatch_S02", #SPW, "Sambre at Salzinnes"
#         "Q_101": "subcatch_S02", #Meuse at St-Mihiel  - SEL ############
#         "Q_16": "subcatch_S06" #Meuse at Borgharen - SEL ############
#         # Roer
#         # Geul 
#         } 
# #%% SPW ########################################
# #SPW - official statistics
# fn_spw = '/p/11208719-interreg/data/spw/statistiques/c_final/spw_statistics.nc' #spw_statistiques doesn't exist anymore
# retlev_spw = xr.open_dataset(fn_spw).load()

# #SPW - raw HOURLY data
# fn_spw_raw = '/p/11208719-interreg/data/spw/Discharge/c_final/hourly_spw_discharges.nc' 
# spw_hourly = xr.open_dataset(fn_spw_raw).load()

# #For the official statistics sheet
# stats_spw = {6228 : "Vesdre at Chaudfontaine",# SPW - old Q_12 - severely hit 2021
#         6621 : "Ambleve at Martinrive",# SPW - old Q_11
#         5921 : "Ourthe at Tabreux",# SPW - old Q_10
#         8221 : "Lesse at Gendron", # SPW - old Q_801
# #        "": "Sambre at Salzinnes", #SPW - old Q_9 - cannot find official statistics
#         9434 : "Semois at Membre"}
# #        "" : "Sambre at Floriffoux"} - cannot find ID in the nc file

# locs_spw = [f'Q_{i}' for i in spw_hourly['id'].values]

# #%%HYDRO ########################################
# fn_hydro = '/p/11208719-interreg/data/hydroportail/c_final/hydro_statistiques_daily.nc'
# retlev_hydro_daily = xr.open_dataset(fn_hydro).load()
# #Meuse a Chooz - Ile Graviat -->  Is it Meuse at Chooz, there are two
# stats_hydro = {0: "Meuse at Chooz",
#                 1: "Meuse at Goncourt"}

# hydro_daily = xr.open_dataset('/p/11208719-interreg/data/hydroportail/c_final/hydro_daily.nc').load()
# #There are two Meuse a Chooz - Ile Graviat is I think Q_4. wflow_id: 1720000002 (also longest time series)

# locs_hydro = {'Q_1011': "Meuse at Goncourt", #hydrofrance
#                 'Q_4': "Meuse at Chooz", #hydrofrance et SPW
#                 }

# #%% Waterschap Limburg ########################################
# wslimburg_daily = xr.open_dataset('/p/11208719-interreg/data/waterschap_limburg/c_final/hydro_D_wl.nc').load()

# #Only locations we have in the csv - IN THE FUTURE WE CAN PICK MORE WITH 
# locs_limburg = {'Q_2004': 26, #Hambeek
#                 'Q_2003': 312, # Vlootbeek
#                 'Q_2001' : 1036 #Geul Meerssen
#                 }

# #No official stats

# #%% Hygon - German partners - official statistics and data missing???
# gpd_hygon = gpd.GeoDataFrame.from_file(r'/p/11208719-interreg/wflow/m_snakecal03/staticgeoms/gauges_hygon.geojson')
# #%% We load the results 
# use_cols = ['time'] + locs_spw + list(locs_hydro.keys()) + list(locs_limburg.keys()) 

#%% We plot the locations in a background map
#We store the DEM
da = mod.staticmaps["wflow_dem"].raster.mask_nodata()
da.attrs.update(long_name="elevation", units="m")
da['lat'].attrs.update(long_name="latitude", units="deg")
da['lon'].attrs.update(long_name="longitude", units="deg")

# create nice colormap
vmin, vmax = da.quantile([0.0, 0.999]).compute()
c_dem = plt.cm.terrain(np.linspace(0.25, 1, 256))
cmap = colors.LinearSegmentedColormap.from_list("dem", c_dem)
norm = colors.Normalize(vmin=vmin, vmax=vmax)
kwargs = dict(cmap=cmap, norm=norm)
#%%

#Plotting
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6,10))
mod.basins.boundary.plot(ax = ax, color='k', lw=1) #The Meuse catchments
da.drop("spatial_ref").plot(ax = ax, **kwargs)
mod.staticgeoms['rivers'].plot(ax = ax, color='b', lw=mod.staticgeoms['rivers']['strord'].values/4, zorder = 1)
#Showing the SPW gauges with data and statistics
mod.staticgeoms['gauges_spw'].plot(ax = ax, color='gold', marker = '^', markersize=25, edgecolor='k', lw=0.5, zorder = 2, label='SPW')
mod.staticgeoms['gauges_spw'].set_index('id').loc[stats_spw.keys()].plot(ax = ax, color='gold', marker = '^', edgecolor = 'r', markersize=35, lw=1.5, zorder = 2)
#Showing the hydro France gauges data and statistics
ax.scatter(x=hydro_daily['x'].values, y=hydro_daily['y'].values, c='seagreen', marker = 's', s=35, edgecolors = 'r', lw=1.5, label='hydroportail')
#Showing the waterschap Limburg gauges data and statistics - no statistics
ax.scatter(x=wslimburg_daily['x'].values, y=wslimburg_daily['y'].values, c='royalblue', marker = 'P', edgecolors = 'k', s=30, lw=0.5, zorder = 2, label='Wat. Limburg')
#Showing the hygon gauges data and statistics - no statistics
gpd_hygon.plot(ax=ax, c='violet', marker = 'd', markersize = 25, edgecolors='k', lw= 0.5, zorder = 3, label = 'hygon')
plt.legend(loc="lower left")
plt.show()
fig.savefig(os.path.join(fn_fig, 'background_data_stations.png'), dpi = 400)

#%% Plotting of the focuse area for June 30
#color=iter(cm.rainbow(np.linspace(0,1,len(shp_catch.keys())+1)))
#color=iter(cm.hsv(np.linspace(0,1,len(shp_catch.keys())+1)))
color=iter(cm.Set1(np.linspace(0,1,len(shp_catch.keys())+1))) #Paired #Set1
w_s = np.linspace(4,1, len(shp_catch.keys()))
w_s = np.array([4.5,2.5,2.5,2.5,2.5,2,2])

#Plotting
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6,10))
mod.basins.boundary.plot(ax = ax, color='k', lw=1) #The Meuse catchments
da.drop('spatial_ref').plot(ax = ax, **kwargs)
mod.staticgeoms['rivers'].plot(ax = ax,  color = 'navy',lw=mod.staticgeoms['rivers']['strord'].values/4, zorder = 1) #color='navy',
#cx.add_basemap(ax, crs=mod.staticgeoms['rivers'].crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik, zoom=8) #, source=cx.providers.Stamen.TerrainBackground,, source=cx.providers.Stamen.TonerLite)
i = 0
for label in shp_catch.keys():
        key = list(shp_catch[label].keys())
        print(key)
        c=next(color)
        print(colors.to_rgb(c))
        #ax.plot(gpd.GeoDataFrame(geometry=mod.staticgeoms[key[0]].set_index('value').loc[shp_catch[label][key[0]]], crs= mod.staticgeoms[key[0]].crs), edgecolor=(colors.to_rgb(c) + (1,)), facecolor=(colors.to_rgb(c) + (0.4,)), linewidth=2, zorder = 3, label=label)
        df = mod.staticgeoms[key[0]]
        df.rename(columns={df.columns[0]:'value'}, inplace = True)
        max = gpd.GeoDataFrame(geometry=df.set_index('value').loc[shp_catch[label][key[0]]], crs= mod.staticgeoms[key[0]].crs).plot(ax = ax, edgecolor=(colors.to_rgb(c) + (1,)), facecolor="None", linewidth=w_s[i], zorder = 3, label=label) #facecolor=(colors.to_rgb(c) + (0.4,))
        #max.legend()
        i += 1
#ax.legend(['River','Meuse'] + list(shp_catch.keys()), loc="lower left", fontsize=8)
ax.set_xlabel('longitude - degrees')
ax.set_ylabel('latitude - degrees')

fig.savefig(os.path.join(fn_fig, 'background_focus_appendix_elevation.png'), dpi = 400)
#%%
#Meuse at Borgharen
gpd.GeoDataFrame(geometry=mod.staticgeoms[shp_catch["Q_16"]].set_index('wflow_'+shp_catch["Q_16"]).loc[float("Q_16".strip('Q_'))], crs= mod.staticgeoms[shp_catch["Q_16"]].crs).plot(ax = ax, edgecolor='r', facecolor="none", linewidth=2, zorder = 3, label="Meuse at Borgharen")
gpd.GeoDataFrame(geometry=mod.staticgeoms['gauges_Sall'].set_index('wflow_id').loc[float("Q_16".strip('Q_'))], crs= mod.staticgeoms[shp_catch["Q_16"]].crs).plot(ax = ax, color='r', zorder = 1)
#Meuse at St Mihiel
gpd.GeoDataFrame(geometry=mod.staticgeoms[shp_catch["Q_101"]].set_index('wflow_'+shp_catch["Q_101"]).loc[float("Q_101".strip('Q_'))], crs= mod.staticgeoms[shp_catch["Q_101"]].crs).plot(ax = ax, edgecolor='chartreuse', facecolor="none",  linewidth=2, zorder = 3, label="Meuse at St-Mihiel")
gpd.GeoDataFrame(geometry=mod.staticgeoms['gauges_Sall'].set_index('wflow_id').loc[float("Q_101".strip('Q_'))], crs= mod.staticgeoms[shp_catch["Q_101"]].crs).plot(ax = ax, color='chartreuse', zorder = 1)
#Lesse at Gendron
gpd.GeoDataFrame(geometry=mod.staticgeoms[shp_catch["Q_801"]].set_index('wflow_'+shp_catch["Q_801"]).loc[float("Q_801".strip('Q_'))], crs= mod.staticgeoms[shp_catch["Q_801"]].crs).plot(ax = ax, edgecolor='b', facecolor="none", linewidth=2, zorder = 3, label="Lesse at Gendron")
gpd.GeoDataFrame(geometry=mod.staticgeoms['gauges_Sall'].set_index('wflow_id').loc[float("Q_801".strip('Q_'))], crs= mod.staticgeoms[shp_catch["Q_801"]].crs).plot(ax = ax, color='b', zorder = 1)
#Vesdre at Chaudfontaine
gpd.GeoDataFrame(geometry=mod.staticgeoms[shp_catch["Q_12"]].set_index('wflow_'+shp_catch["Q_12"]).loc[float("Q_12".strip('Q_'))], crs= mod.staticgeoms[shp_catch["Q_12"]].crs).plot(ax = ax, edgecolor='darkviolet', facecolor="none",  linewidth=2, zorder = 3, label="Vesdre at Chaudfontaine")
gpd.GeoDataFrame(geometry=mod.staticgeoms['gauges_Sall'].set_index('wflow_id').loc[float("Q_12".strip('Q_'))], crs= mod.staticgeoms[shp_catch["Q_12"]].crs).plot(ax = ax, color='darkviolet', zorder = 1)
ax.legend(['River','Meuse', "Meuse at Borgharen", "Meuse at St-Mihiel", "Lesse at Gendron", "Vesdre at Chaudfontaine"], loc="lower left", fontsize=8)
plt.show()
fig.savefig(os.path.join(fn_fig, 'background_focus_June30.png'), dpi = 400)
#%% We perform 


#SPW

#SPW - Statistiques

#HYDRO

#Hydro - Statistiques