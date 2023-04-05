# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 12:29:31 2022

@author: bouaziz
"""

import hydromt
import xarray as xr
import numpy as np
import os
import xarray as xr
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import h5py
import hdf5plugin
from func_plot_signature_joost import plot_hydro
from func_plot_signature_joost import plot_signatures

#%%

stations_dic = {
    "Meuse at Goncourt" : 1011,
    "Mouzon at Circourt-sur-Mouzon" : 1013, 
    "Vair at Soulosse-sous-Saint-Elophe" : 1016, 
    "Meuse at Saint-Mihiel" : 101, 
    "Meuse at Stenay" : 3,
    "Bar at Cheveuges" : 41, 
    "Vence at Francheville" : 42, 
    "Sormonne at Belval" : 43, 
    "Semois at Membre-Pont" : 5, 
    "Semois at Sainte-Marie" : 503, 
    "Vierre at Straimont" : 501,
    "Chiers at Carignan" : 201, 
    "Chiers at Longlaville" : 203, 
    "Crusnes at Pierrepont" : 206, 
    "Ton at Écouviez" : 207, 
    "Loison at Han-les-Juvigny" : 209,
    "Viroin at Treignes" : 6,
    "Meuse at Chooz" : 4,
    "Lesse at Daverdisse" : 802, 
    "Lhomme at Jemelle" : 803, 
    "Lesse at Gendron" : 801,
    "Hermeton at Hastiere" : 701, 
    "Bocq at Yvoir" : 702, 
    "Molignee at Warnant" : 703, 
    "Hoyoux at Modave" : 704, 
    "Ourthe Occidentale at Ortho" : 1002, 
    "Ourthe Orientale at Mabompre" : 1003, 
    "Ourthe at Tabreux" : 10, 
    "Hantes at Wiheries" : 903, 
    "Sambre at Salzinnes" : 9,
    "Mehaigne at Huccorgne" : 13,
    "Meuse at Amay" : 1401,
    "Ambleve at Martinrive" : 11, 
    "Vesdre at Chaudfontaine" : 12,
    "Meuse at Borgharen" : 16}

stations = [1011, 1013, 1016, 
                   101, 3, 
             41, 42, 43, 
             5, 503, 501,
             201, 203, 206, 207, 209,
             6,
             4,
             802, 803, 801,
             701, 702, 703, 704, 
             1002, 1003, 10, 
             903, 9,
             13,
             1401,
             11, 
             12,
             16]

stations_s01= [1016, 1013, 1011,41, 42, 43, 503, 501,6,802, 803, 701, 702, 703, 704, 903,1002, 1003, 11,12, 13, 203, 206, 209, 207]
stations_be = [5, 6, 503, 501, 801, 802, 803, 701, 702, 703, 704, 903,1002, 1003,10, 11,12, 13,1401]

#%% obs data

#all daily
ds_obs = xr.open_dataset(r'd:\Promotie\Data\qobs_xr.nc')
# ds_obs = xr.open_dataset(r"p:\11205237-grade\wflow\wflow_meuse_julia\_obs\qobs_xr.nc")

#belgian hourly 
qobs_h = xr.open_dataset(r"d:\GRADE\Kennisontwikkeling\Products_2021\era5_Martijn\Meuse\qobs_hourly_belgian_catch.nc")
# qobs_h = xr.open_dataset(r"p:\11205237-grade\wflow\wflow_meuse_julia\_obs\qobs_hourly_belgian_catch.nc")
qobs_h = qobs_h.rename({"catchments":"stations"})

#french hourly data
qobs_h_fr = xr.open_dataset(r"p:\11205237-grade\wflow\wflow_meuse_julia\data_hourly_hydroeau\FR-Hydro-hourly-2005_2022.nc")
qobs_h_fr = qobs_h_fr.rename({"Q":"Qobs_m3s", "wflow_id":"stations"})

#borgharen
stpieter = pd.read_csv(r"p:\11205237-grade\wflow\wflow_meuse_julia\data_rws\20221205_ST_PIETER\05_OUTPUT\ST_PIETER.csv", index_col=0, parse_dates=True, header=0)
stpieter = stpieter.resample("H").mean()
stpieter_xr = stpieter.to_xarray()
stpieter_xr = stpieter_xr.rename({"value":"Qobs_m3s"})
stpieter_xr = stpieter_xr.rename({"timestamp":"time"})
stpieter_xr = stpieter_xr.assign_coords({"stations": 16}).expand_dims("stations")
stpieter_xr["Qobs_m3s"].sel(time=slice(None, "2019-12-31")).plot()

qobs_h = xr.merge([qobs_h, qobs_h_fr, ] )

qobs_h = xr.merge([qobs_h, stpieter_xr.sel(time=slice(None, "2019-12-31"))])



#%% modeled data 

Folder_p = r"p:\11208719-interreg\wflow"

model_runs = {
    "default": {"case":"a_floodplain1d", 
                "folder": "run_default"},
    
    "rz20": {"case": "b_rootzone",
             "folder": "run_rootingdepth_rp_20"},
    
}


### prepare dataset to make plots
colors = [
    '#a6cee3','#1f78b4',
    '#b2df8a','#33a02c',
    '#fb9a99','#e31a1c',
    '#fdbf6f','#ff7f00',
    '#cab2d6','#6a3d9a',
    '#ffff99','#b15928']

runs_dict = {}

for key in model_runs.keys():
    print(key)
    case = model_runs[key]["case"]
    folder = model_runs[key]["folder"] 
    runs_dict[key] = pd.read_csv(os.path.join(Folder_p, case, folder, "output.csv"), index_col=0, header=0, parse_dates=True)


plot_colors = colors[:len(runs_dict)]

#%%

caserun = "rootzone_rp20"

Folder_plots = r"d:\interreg\Plots" + "\\" + f"{caserun}"

if not os.path.exists(Folder_plots):
    os.mkdir(Folder_plots)

#make dataset
variables = ['Q','P', 'EP']    
    
start = '2005-01-01 01:00:00'
end = '2017-12-31 23:00:00'
rng = pd.date_range(start, end, freq="H")

S = np.zeros((len(rng), len(stations), len(list(runs_dict.keys())+["HBV", "Obs."])))
v = (('time', 'stations', 'runs'), S)
h = {k:v for k in variables}

ds = xr.Dataset(
        data_vars=h, 
        coords={'time': rng,
                'stations': stations,
                'runs': list(list(runs_dict.keys())+["HBV", "Obs."])})
ds = ds * np.nan

for key in runs_dict.keys(): 
    print(key)
    #fill dataset with model and observed data
    ds['Q'].loc[dict(runs = key)] = runs_dict[key][['Q_' + sub for sub in list(map(str,stations))]].loc[start:end]    
    # ds['H'].loc[dict(runs = key)] = runs_dict[key][['H_' + sub for sub in list(map(str,stations))]].loc[start:end]
    
ds['P'].loc[dict(runs = "Obs.")] = runs_dict[key][['P_' + sub for sub in list(map(str,stations))]].loc[start:end]
ds['EP'].loc[dict(runs = "Obs.")] = runs_dict[key][['EP_' + sub for sub in list(map(str,stations))]].loc[start:end]

#fill obs data
ds['Q'].loc[dict(time = qobs_h.time.loc[start:end], stations=qobs_h.stations.values, runs = 'Obs.', )] = qobs_h["Qobs_m3s"].loc[dict(time=slice(start,end))]



#make plots
start_long = '2006-01-01 01:00:00'
end_long =  '2017-12-31 23:00:00'
start_1 =  '2010-11-01'
end_1 = '2011-03-01'
start_2 =  '2011-03-01'
end_2 =  '2011-10-31'
start_3 =  '2015-01-01'
end_3 = '2015-12-31'

for station_name, station_id in stations_dic.items():
    if station_id in qobs_h.stations.values:    
        print(station_name)
        runs_sel = list(runs_dict.keys()) 
        plot_colors = colors[:len(runs_dict)]
        dsq = ds.sel(stations = station_id).sel(time = slice('2006-01-01', "2015-12-31"), runs=runs_sel + ["Obs."])#.dropna(dim='time')
        #plot hydro
        plot_hydro(dsq, start_long, end_long, start_1, end_1, start_2, end_2, start_3, end_3, runs_sel, plot_colors, Folder_plots, station_name, save=True)
        plt.close()
           
        #make plot using function
        #dropna for signature calculations. 
        #start later for for warming up
        dsq = ds['Q'].sel(stations = station_id).sel(time = slice('2006-01-01', "2015-12-31"), runs=runs_sel + ["Obs."]).to_dataset().dropna(dim='time')
        #TODO: somehow xr.infer_freq(dsq.time) does not work for Borgharen..... 
        plot_signatures(dsq, runs_sel + ["Obs."], plot_colors, Folder_plots, station_name, save=True, window=7*24)
        plt.close()
    else:
        print(f"no obs data for {station_name}")




