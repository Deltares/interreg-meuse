# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 12:29:31 2022

@author: bouaziz
"""
#%%
import hydromt
from hydromt_wflow import WflowModel
import xarray as xr
import numpy as np
import os
import xarray as xr
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import geopandas as gpd
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

#%% 



#%% obs data

#all daily
ds_obs = xr.open_dataset(r'd:\Promotie\Data\qobs_xr.nc')
# ds_obs = xr.open_dataset(r"p:\11208719-interreg\data\observed_streamflow_grade\qobs_xr.nc")

#belgian hourly 
qobs_h = xr.open_dataset(r"d:\GRADE\Kennisontwikkeling\Products_2021\era5_Martijn\Meuse\qobs_hourly_belgian_catch.nc")
# qobs_h = xr.open_dataset(r"p:\11208719-interreg\data\observed_streamflow_grade\qobs_hourly_belgian_catch.nc")
qobs_h = qobs_h.rename({"catchments":"stations"})

#french hourly data
qobs_h_fr = xr.open_dataset(r"p:\11208719-interreg\data\observed_streamflow_grade\FR-Hydro-hourly-2005_2022.nc")
qobs_h_fr = qobs_h_fr.rename({"Q":"Qobs_m3s", "wflow_id":"stations"})

qobs_h = xr.merge([qobs_h, qobs_h_fr])


#all daily
ds_obs_spw = xr.open_dataset(r"p:\11208186-spw\Data\Debits\Q_jour\hydro_daily.nc")
ds_obs_spw = ds_obs_spw.transpose()


#%% modeled data 

Folder_p = r"p:\11208719-interreg\wflow"

#routing runs
model_runs = {
    "kinematic": {"case":"d_manualcalib", 
                "folder": "run_manualcalib_daily_eobs24_kinematic"},
    
    "loc.iner": {"case": "d_manualcalib",
             "folder": "run_manualcalib_daily_eobs24_1d"},
    
    "loc.iner.flpl1d": {"case":"d_manualcalib", 
                "folder": "run_manualcalib_daily_eobs24"},
    
    "loc.iner1d2d": {"case": "d_manualcalib",
             "folder": "run_manualcalib_daily_eobs24_1d2d"},
    
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

caserun = "routing"   
Folder_plots = r"d:\interreg\Plots" + "\\" + f"{caserun}"

if not os.path.exists(Folder_plots):
    os.mkdir(Folder_plots)

#make dataset
variables = ['Q'] #,'P', 'EP']    
    
start = '1980-01-01'
end = '2020-12-30'
rng = pd.date_range(start, end)

S = np.zeros((len(rng), len(stations), len(list(runs_dict.keys())+[ "Obs."])))
v = (('time', 'index', 'runs'), S)
h = {k:v for k in variables}

ds = xr.Dataset(
        data_vars=h, 
        coords={'time': rng,
                'index': stations,
                'runs': list(list(runs_dict.keys())+[ "Obs."])})
ds = ds * np.nan

for key in runs_dict.keys():
    print(key)
    #fill dataset with model and observed data
    ds['Q'].loc[dict(runs = key)] = runs_dict[key][['Q_' + sub for sub in list(map(str,stations))]].loc[start:end]    
    
# ds['P'].loc[dict(runs = "Obs.")] = runs_dict[key][['P_' + sub for sub in list(map(str,stations))]].loc[start:end]
# ds['EP'].loc[dict(runs = "Obs.")] = runs_dict[key][['EP_' + sub for sub in list(map(str,stations))]].loc[start:end]

#fill obs data
# ds['Q'].loc[dict(runs = 'Obs.', time = ds_obs_spw.time.loc[start:end], index=stations)] = ds_obs_spw["Q"].sel(time=slice(start,end), id=stations).rename({"index":"id_spw"}).rename({"id":"index"}).transpose("time", "index")
#fill obs data
ds['Q'].loc[dict(runs = 'Obs.', time = ds_obs.time.loc[start:end])] = ds_obs["Qobs"].loc[start:end]


#make plots
start_long = '1992-01-01'
end_long =  '2015-12-31'
start_1 =  '2010-11-01'
end_1 = '2011-03-01'
start_2 =  '2011-03-01'
end_2 =  '2011-10-31'
start_3 =  '2015-01-01'
end_3 = '2015-12-31'

for station_name, station_id in stations_dic.items():
    print(station_id, station_name)
    runs_sel = list(runs_dict.keys()) 
    plot_colors = colors[:len(runs_dict)]
    dsq = ds.sel(index = station_id).sel(time = slice('1993-01-01', None), runs = runs_sel + ["Obs."])#.dropna(dim='time')
    #plot hydro
    plot_hydro(dsq, start_long, end_long, start_1, end_1, start_2, end_2, start_3, end_3, runs_sel, plot_colors, Folder_plots, f"{station_name}_{station_id}" , save=True)
    plt.close()
       
    #make plot using function
    #dropna for signature calculations. 
    #start later for for warming up
    dsq = ds['Q'].sel(index = station_id, runs = runs_sel + ["Obs."]).sel(time = slice('2000-01-01', '2020-12-31')).to_dataset().dropna(dim='time')
    if len(dsq.time)>366*2:
        plot_signatures(dsq, runs_sel, plot_colors, Folder_plots, f"{station_name}_{station_id}" , save=True, window=7)
        plt.close()



