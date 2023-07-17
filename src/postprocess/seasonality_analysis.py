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

#%% We import the modelled data
Folder_start = "/p/11208719-interreg"
model_wflow = "f_spwgauges"
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

fn_runs = glob.glob(os.path.join(Folder_p, folder, '*', 'output.csv'))
date_parser = lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S")

model_runs = {
    "r1": {"case":folder, 
            "folder": "r1i1p5f1"},
    "r2": {"case":folder, 
            "folder": "r2i1p5f1"},
    "r3": {"case":folder, 
            "folder": "r3i1p5f1"},
    "r4": {"case":folder, 
            "folder": "r4i1p5f1"}, 
    "r5": {"case":folder, 
            "folder": "r5i1p5f1"}, 
    "r6": {"case":folder, 
            "folder": "r6i1p5f1"}, 
    "r7": {"case":folder, 
            "folder": "r7i1p5f1"}, 
    "r8": {"case":folder, 
            "folder": "r8i1p5f1"}, 
    "r9": {"case":folder, 
            "folder": "r9i1p5f1"}, 
    "r10": {"case":folder, 
            "folder": "r10i1p5f1"}, 
    "r11": {"case":folder, 
            "folder": "r11i1p5f1"}, 
    "r12": {"case":folder, 
            "folder": "r12i1p5f1"}, 
    "r13": {"case":folder, 
            "folder": "r13i1p5f1"}, 
    "r14": {"case":folder, 
            "folder": "r14i1p5f1"}, 
    "r15": {"case":folder, 
            "folder": "r15i1p5f1"}, 
    "r16": {"case":folder, 
            "folder": "r16i1p5f1"}, 
} 
#%%Important locations
locs = {'Q_1011': "Meuse at Goncourt", #hydrofrance
        'Q_4': "Meuse at Chooz", #hydrofrance et SPW
        'Q_16': "Meuse at Borgharen", #/St Pieter -- station recalculée Discharge at Kanne + discharge at St Pieter 
        'Q_12': "Vesdre at Chaudfontaine",# SPW - severely hit 2021
        'Q_11': "Ambleve at Martinrive",# SPW 
        'Q_10': "Ourthe at Tabreux",# SPW 
        'Q_801': "Lesse at Gendron", # SPW
        "Q_9": "Sambre at Salzinnes", #SPW
        "Q_101": "Meuse at St-Mihiel" 
        # Roer
        # Geul 
        } 

use_cols = ['time'] + list(locs.keys()) 
#%% Other parameters for POT
thr = {'Q_1011': 35, #hydrofrance
        'Q_4': 600, #hydrofrance et SPW
        'Q_16': 1000, #/St Pieter -- station recalculée Discharge at Kanne + discharge at St Pieter 
        'Q_12': 80,# SPW - severely hit 2021
        'Q_11': 130,# SPW 
        'Q_10': 150,# SPW 
        'Q_801': 160, # SPW
        "Q_9": 250, #SPW
        "Q_101": 150 ######################
        # Roer
        # Geul 
        } 

r_decl = {'Q_1011': '7D', #hydrofrance
        'Q_4': '7D', #hydrofrance et SPW
        'Q_16': '7D', #/St Pieter -- station recalculée Discharge at Kanne + discharge at St Pieter 
        'Q_12': '7D',# SPW - severely hit 2021
        'Q_11': '7D',# SPW 
        'Q_10': '7D',# SPW 
        'Q_801': '7D', # SPW
        "Q_9": '7D', #SPW
        "Q_101": '7D',
        # Roer
        # Geul 
        } 

shp_catch = {'Q_1011': "subcatch_S01", #hydrofrance
        'Q_4': "subcatch_S04", #hydrofrance et SPW
        'Q_16': 'subcatch_S06', #/St Pieter -- station recalculée Discharge at Kanne + discharge at St Pieter 
        'Q_12': "subcatch_S01",# SPW - severely hit 2021
        'Q_11': "subcatch_S01",# SPW 
        'Q_10': "subcatch_S02",# SPW 
        'Q_801': "subcatch_S02", # SPW
        "Q_9": "subcatch_S02", #SPW
        "Q_101": "subcatch_S02"
        # Roer
        # Geul 
        } 
#%%
# Storing the results in one location 
runs_dict = {}
for key in model_runs.keys():
    print(key)
    case = model_runs[key]["case"]
    ens = model_runs[key]["folder"] 
    runs_dict[key] = pd.read_csv(os.path.join(Folder_p, case, ens, "output.csv"), index_col=['time'], header=0, usecols = use_cols, parse_dates=['time'], date_parser = date_parser)
#%%
for station in locs.keys():
    print(station)
    #station = 'Q_101'
    #
    print(station)

    area_m2 = np.float(gpd.GeoDataFrame(geometry=mod.staticgeoms[shp_catch[station]].set_index('wflow_'+shp_catch[station]).loc[float(station.strip('Q_'))], crs= mod.staticgeoms[shp_catch[station]].crs).to_crs(3044).area.values) 
    area_km2 = area_m2 /1000000

    print(f"{locs[station]} has an area in km2 of: {area_km2}")

    #We extract the peaks - to get 2 extremes per year? 1040*2
    ts_pot=pd.DataFrame()
    for key in model_runs.keys():
            print(key)
            ts_key = runs_dict[key][station]

            #We perform the EVA - POT - GP 
            peaks = pyex.get_extremes(ts_key, method = 'POT', extremes_type="high", threshold = thr[station], r=r_decl[station])
            peaks = peaks.reset_index(drop=False)
            peaks['member'] = key
            ts_pot = pd.concat([ts_pot, peaks], axis = 0, ignore_index = True)
            print(len(ts_pot))
    print('Average number of events per year:', len(ts_pot)/1040)

    #We take the top X events and check when they happened
    top_n = [10, 50, 104, 500, 1040]
    top_peaks = ts_pot.sort_values(by = station, axis = 0, ascending=False).reset_index(drop=True)
    top_peaks['month'] = top_peaks['time'].dt.month

    #
    for n in top_n:
        #PLotting the histogram of the highest values
        fig = plt.figure()
        #I shoudl use ax.bar instead()
        plt.hist(top_peaks['month'].iloc[0:n].values, bins=np.arange(14)-0.5, edgecolor = 'black', color='blue', stacked = True, density = True)
        plt.xticks(range(13))
        plt.title(f'Top {n} POT events - {locs[station]}')
        plt.xlim(0.5,12.5)
        plt.show()
        fig.savefig(os.path.join(fn_fig, f'{station}_{dt}_top_hist_{n}_POT_events.png'), dpi=400)

    #We split this between summer and winter event
    m_winter = [10,11,12,1,2,3]
    m_summer = [4,5,6,7,8,9]

    tops = {'winter' : top_peaks.where(top_peaks.loc[:,'time'].dt.month.isin(m_winter)).dropna(how='all').reset_index(drop=True),
            'summer': top_peaks.where(top_peaks.loc[:,'time'].dt.month.isin(m_summer)).dropna(how='all').reset_index(drop=True)}

    #We plot the footprint of the top events
    if dt == 'daily':
        acc_time = int(r_decl[station].strip('D'))
    if dt == 'hourly':
        acc_time = int(r_decl[station].strip('D'))*24

    msk = mod.staticmaps['wflow_subcatch']==True
    msk_catch = mod.staticmaps['wflow_'+shp_catch[station]]== int(station.strip('Q_'))

    area_m2 = np.float(gpd.GeoDataFrame(geometry=mod.staticgeoms[shp_catch[station]].set_index('wflow_'+shp_catch[station]).loc[float(station.strip('Q_'))], crs= mod.staticgeoms[shp_catch[station]].crs).to_crs(3857).area.values) 
    area_km2 = area_m2 /1000000
    print(f"{locs[station]} has an area in km2 of: {area_km2}")

    for case_fig in tops.keys():
        print(case_fig)
        sub_tops = tops[case_fig]

        fig, axs = plt.subplots(ncols=5, nrows=2, figsize=(10,6))
        cmap=plt.get_cmap("viridis", 11)
        for ax, i in zip(axs.reshape(-1), np.arange(0,np.min([len(sub_tops), 11]),1)):
            print(i)
            folder = str(model_runs[sub_tops.loc[i,'member']]['folder'])
            year = str(sub_tops.loc[i, 'time'].year)
            year_bef = str(sub_tops.loc[i, 'time'].year - 1)
            if dt == 'daily':
                pr_fn = os.path.join(Folder_start, 'data', 'racmo', 'members_bias_corrected', 'c_wflow', dt, folder, f'ds_merged_{year}.nc')
            if dt == 'hourly':
                pr_fn = os.path.join(Folder_start, 'data', 'racmo', 'members_bias_corrected', 'c_wflow','corrected', dt, folder, f'ds_merged_{year}.nc')


            if sub_tops.loc[i, 'time'].year - 1 < 1950:
                pr = xr.open_dataset(pr_fn)
            else:
                if dt == 'daily':
                        pr_fn_bef = os.path.join(Folder_start, 'data', 'racmo', 'members_bias_corrected', 'c_wflow', dt, folder, f'ds_merged_{year_bef}.nc')
                if dt == 'hourly':
                        pr_fn_bef = os.path.join(Folder_start, 'data', 'racmo', 'members_bias_corrected', 'c_wflow','corrected', dt, folder, f'ds_merged_{year_bef}.nc')
                
                pr = xr.open_mfdataset([pr_fn, pr_fn_bef])

            msk['lat'] = pr['lat']
            msk['lon'] = pr['lon']
            msk_catch['lat'] = pr['lat']
            msk_catch['lon'] = pr['lon']

            ds_i = xr.where(msk==True, pr['precip'].rolling(time=acc_time, center=False).sum().sel(time=sub_tops.loc[i, 'time']),np.nan)
        
            if i ==0:
                seas = ds_i
            else:
                seas = xr.concat([seas, ds_i], "time")

            #Accumulated rainfall
            catch_rain = ds_i.where(msk_catch==True).mean(dim=('lat','lon')).values

            #fig, ax = plt.subplots(1, 1, figsize = (12,12))
            p = ds_i.plot(ax = ax, cmap = cmap, vmin = 10, vmax = 300, add_colorbar=False, yticks=[], xticks=[])
            mod.basins.boundary.plot(ax = ax, color='k', lw=1)
            gpd.GeoDataFrame(geometry=mod.staticgeoms[shp_catch[station]].set_index('wflow_'+shp_catch[station]).loc[float(station.strip('Q_'))], crs= mod.staticgeoms[shp_catch[station]].crs).plot(ax = ax, edgecolor='r', facecolor="none", zorder = 3)
            mod.staticgeoms['rivers'].plot(ax = ax, color='k', lw=0.5, zorder = 2)
            ax.text(x = 0.05, y = 0.1, s =  f'{catch_rain:.1f} mm', transform = ax.transAxes, fontsize='xx-small')
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_title('month: '+str(sub_tops.loc[i, 'time'].month))
        cax = plt.axes([0.90, 0.1, 0.01, 0.8])
        plt.colorbar(p, cax = cax, label='cum. precip. [mm]', extend='max')
        plt.show()
        fig.savefig(os.path.join(fn_fig, f'{station}_{dt}_top_events_{case_fig}.png'), dpi=400)

        #Calculating normalized plots
        seas_norm_meuse = seas / seas.max(dim=('lat','lon'))
        seas_norm_catch = seas / seas.where(msk_catch==True).max(dim=('lat','lon'))
        
        fig = plt.figure()
        g = seas_norm_meuse.plot(x="lon", y="lat", col="time", col_wrap=5, cmap='Reds', vmin = 0, vmax = 1)
        for i, ax in enumerate(g.axs.flat):
                try:
                        gpd.GeoDataFrame(geometry=mod.staticgeoms[shp_catch[station]].set_index('wflow_'+shp_catch[station]).loc[float(station.strip('Q_'))], crs= mod.staticgeoms[shp_catch[station]].crs).plot(ax = ax, edgecolor='k', facecolor="none", zorder = 3)
                        ax.set_title(str(str(sub_tops.loc[i, 'time']).split(' ')[0] + ' '+ sub_tops.loc[i, 'member']))
                except:
                        continue
        plt.savefig(os.path.join(fn_fig, f'{station}_{dt}_top_events_{case_fig}_normalized_meuse.png'), dpi=400)
        plt.show()

        fig, axs = plt.subplots(ncols=1)
        g=seas_norm_meuse.mean(dim='time').drop('spatial_ref').plot(cmap='Reds', vmin = 0, vmax = 1)
        gpd.GeoDataFrame(geometry=mod.staticgeoms[shp_catch[station]].set_index('wflow_'+shp_catch[station]).loc[float(station.strip('Q_'))], crs= mod.staticgeoms[shp_catch[station]].crs).plot(ax = axs, edgecolor='k', facecolor="none", zorder = 3)
        plt.show()
        fig.savefig(os.path.join(fn_fig, f'{station}_{dt}_top_events_{case_fig}_normalized_meuse_mean.png'), dpi=400)

        fig = plt.figure()
        g=seas_norm_catch.plot(x="lon", y="lat", col="time", col_wrap=5, cmap='Reds', vmin = 0, vmax = 1)
        for i, ax in enumerate(g.axs.flat):
                try:
                        gpd.GeoDataFrame(geometry=mod.staticgeoms[shp_catch[station]].set_index('wflow_'+shp_catch[station]).loc[float(station.strip('Q_'))], crs= mod.staticgeoms[shp_catch[station]].crs).plot(ax = ax, edgecolor='k', facecolor="none", zorder = 3)
                        ax.set_title(str(str(sub_tops.loc[i, 'time']).split(' ')[0] + ' '+ sub_tops.loc[i, 'member']))
                except:
                        continue
        plt.savefig(os.path.join(fn_fig, f'{station}_{dt}_top_events_{case_fig}_normalized_catch.png'), dpi=400)
        plt.show()

        fig, axs = plt.subplots(ncols=1)
        g=seas_norm_catch.mean(dim='time').drop('spatial_ref').plot(cmap='Reds', vmin = 0, vmax = 1)
        gpd.GeoDataFrame(geometry=mod.staticgeoms[shp_catch[station]].set_index('wflow_'+shp_catch[station]).loc[float(station.strip('Q_'))], crs= mod.staticgeoms[shp_catch[station]].crs).plot(ax = axs, edgecolor='k', facecolor="none", zorder = 3)
        plt.show()
        fig.savefig(os.path.join(fn_fig, f'{station}_{dt}_top_events_{case_fig}_normalized_catch_mean.png'), dpi=400)

#     #We get the wave form
#     for case_fig in tops.keys():
#         sub_tops = tops[case_fig]

#         waves = pd.DataFrame(index=np.arange(0, 2*acc_time+1,1))
#         norm_waves = pd.DataFrame(index=np.arange(0, 2*acc_time+1,1))
#         for i in np.arange(0,np.min([len(sub_tops), 11]),1):
#             print(i)
#             beg = sub_tops.loc[i,'time'] - timedelta(days=acc_time)
#             end = sub_tops.loc[i,'time'] + timedelta(days=acc_time)
#             ts_key = runs_dict[sub_tops.loc[i,'member']][station][beg:end]
#             norm_ts = ts_key/ts_key.max()
#             waves = pd.concat([waves, ts_key.reset_index(drop=True)], axis=1, ignore_index=True)
#             norm_waves = pd.concat([norm_waves, norm_ts.reset_index(drop=True)], axis=1, ignore_index=True)

#         fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(10,6))
#         axs[0].plot(waves, 'grey', lw=1)
#         axs[0].plot(waves.median(axis=1), 'red', lw=1.5)
#         axs[1].plot(norm_waves, 'grey', lw=1)
#         axs[1].plot(norm_waves.median(axis=1), 'red', lw=1.5)
#         plt.title(case_fig)
#         axs[0].grid('--', lw=0.5)
#         axs[1].grid('--', lw=0.5)
#         fig.savefig(os.path.join(fn_fig, f'{station}_waves_{case_fig}.png'), dpi=400)

# %%
