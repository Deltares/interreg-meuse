# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 12:29:31 2022

@author: bouaziz
"""
#%%
import hydromt
import xarray as xr
import numpy as np
import os
import xarray as xr
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from hydromt_wflow import WflowModel
# import h5py
# import hdf5plugin
from func_plot_signature_joost import plot_hydro
from func_plot_signature_joost import plot_signatures


#%% make dic of all sources 

source_dic = {

    "spw":{"coord":"spw_gauges_spw",
            "fn":"spw_qobs_hourly", #entry in datacatalog 
            "stations": [7319, 5921, 6228, 6621, 9434, 9021, 8221],},

    "wl":{"coord":"wl_gauges_waterschaplimburg",
            "fn":"wl_qobs_hourly", 
            "stations": [1036, 1030],},

    "hp":{"coord":"hp_gauges_hydroportail",
            "fn":"hp_qobs_hourly", 
            "stations": [1022001001, 1720000001],},

    "rwsinfo":{"coord":"rwsinfo_gauges_rwsinfo",
            "fn":"rwsinfo_qobs_hourly", 
            "stations": [16],},    

    "france":{"coord":"france_gauges_france",
            "fn":"france_qobs_hourly", 
            "stations": [12220010],},    

    # #not available for hourly data!
    # "hygon":{"coord":"hygon_gauges_hygon",
    #         "fn":"hygon_qobs_hourly", #entry in datacatalog 
    #         "stations": [],},

                     }

#%% read model 

root = r"p:\11208719-interreg\wflow\o_rwsinfo"
config_fn = "run_rwsinfo.toml"
yml = r"p:\11208719-interreg\data\data_meuse.yml"
mod = WflowModel(root = root, config_fn=config_fn, data_libs=["deltares_data", yml], mode = "r")


#%% modeled data 

Folder_p = r"p:\11208719-interreg\wflow"

model_runs = {

    #rwsinfo 
    "mod": {"case": "o_rwsinfo",
             "folder": "run_rwsinfo"},

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
    # runs_dict[key] = pd.read_csv(os.path.join(Folder_p, case, folder, "output.csv"), index_col=0, header=0, parse_dates=True)
    runs_dict[key] = xr.open_dataset(os.path.join(Folder_p, case, folder, "output_scalar.nc"))


plot_colors = colors[:len(runs_dict)]

#%%


caserun = "historic_hourly_o_rwsinfo"

Folder_plots = r"d:\interreg\Plots" + "\\" + f"{caserun}"

if not os.path.exists(Folder_plots):
    os.mkdir(Folder_plots)


#%% make dataset 

var = "Q"

start = '2005-01-01 01:00:00'
end = '2017-12-31 23:00:00'
rng = pd.date_range(start, end, freq="H")

for source in source_dic:
    print(source)
    coord_name = source_dic[source]["coord"]
    fn = source_dic[source]["fn"]
    stations_sel = source_dic[source]["stations"]

    #get observed ds
    if source == "spw":
        #if spw make sure station_name becomes a coordinate instead of a data variable
        ds_obs = mod.data_catalog.get_geodataset(fn, variables = ["Q", "Nom"], time_tuple=(start, end))
        #add station name to coordinates:
        ds_obs = ds_obs.assign_coords({"station_name" : ("wflow_id", ds_obs.Nom.values)})
        ds_obs = ds_obs["Q"]
    else:
        ds_obs = mod.data_catalog.get_geodataset(fn, variables = ["Q"], time_tuple=(start, end))
    ds_obs_sel = ds_obs.sel(wflow_id = stations_sel)
    ds_obs_sel = ds_obs_sel.assign_coords({"runs":"Obs."}) .expand_dims("runs")
    #rename wflow_id to stations
    ds_obs_sel = ds_obs_sel.rename({"wflow_id":"stations"})

    
    for key in runs_dict.keys(): 
        print(key)
        ds_mod = runs_dict[key]
        ds_mod_sel = ds_mod.sel({f"{var}_{coord_name}" : list(map(str, stations_sel))}).sel(time = slice(start, end))
        #rename Q_source to Q 
        ds_mod_sel = ds_mod_sel.rename({f"{var}_{source}":f"{var}"})[f"{var}"]
        #chunk 
        ds_mod_sel = ds_mod_sel.chunk({f"{var}_{coord_name}":len(stations_sel)}) #  (dict(Q_spw_gauges_spw=3))
        #add runs coord 
        ds_mod_sel = ds_mod_sel.assign_coords({"runs":key}).expand_dims("runs")
        #rename coord to "station"
        ds_mod_sel = ds_mod_sel.rename({f"{var}_{coord_name}":"stations"})
        #make sure stations is int instead of str
        ds_mod_sel["stations"] = list(map(int, list(ds_mod_sel["stations"].values)))

    #combine obs and runs in one dataset 
    ds = xr.concat([ds_obs_sel, ds_mod_sel], dim = "runs").to_dataset()
    ds = ds.load()


    #make plots
    start_long = '2006-01-01 01:00:00'
    end_long =  '2017-12-31 23:00:00'
    start_1 =  '2010-11-01'
    end_1 = '2011-03-01'
    start_2 =  '2011-03-01'
    end_2 =  '2011-10-31'
    start_3 =  '2015-01-01'
    end_3 = '2015-12-31'

    for station_id in ds.stations.values:
        print(station_id)
        try: 
            station_name = ds["station_name"].sel(stations=station_id).values
        except:
            station_name = str(station_id)
        if len(ds["Q"].sel(runs = "Obs.", stations = station_id)) > 0:    
            # print(station_name)
            runs_sel = list(runs_dict.keys()) 
            plot_colors = colors[:len(runs_dict)]
            dsq = ds.sel(stations = station_id).sel(time = slice('2006-01-01', "2017-12-31"), runs=runs_sel + ["Obs."])#.dropna(dim='time')
            #load in memory 
            # dsq = dsq.load()
            #plot hydro
            plot_hydro(dsq, start_long, end_long, start_1, end_1, start_2, end_2, start_3, end_3, runs_sel, plot_colors, Folder_plots, f"{source}_{station_name}_{station_id}", save=True)
            plt.close()
            
            #make plot using function
            #dropna for signature calculations. 
            #start later for for warming up
            dsq = ds['Q'].sel(stations = station_id).sel(time = slice('2006-01-01', "2017-12-31"), runs=runs_sel + ["Obs."]).to_dataset().dropna(dim='time')
            #load in memory 
            # dsq = dsq.load()
            plot_signatures(dsq, runs_sel + ["Obs."], plot_colors, Folder_plots, f"{source}_{station_name}_{station_id}", save=True, window=7*24)
            plt.close()
        else:
            print(f"no obs data for {station_name}")



