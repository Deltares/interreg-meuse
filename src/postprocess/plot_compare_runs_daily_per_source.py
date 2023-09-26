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


#%% make dic of all sources 

source_dic = {

    "spw":{"coord":"spw_gauges_spw",
            "fn":"spw_qobs_daily", #entry in datacatalog 
            "stations": [7319, 5921, 6228, 6621, 9434, 9021, 8221],},

    "wl":{"coord":"wl_gauges_waterschaplimburg",
            "fn":"wl_qobs_daily", 
            "stations": [1036, 1030],},

    "hp":{"coord":"hp_gauges_hydroportail",
            "fn":"hp_qobs_daily", 
            "stations": [1022001001, 1720000001],},

    "rwsinfo":{"coord":"rwsinfo_gauges_rwsinfo",
            "fn":"rwsinfo_qobs_daily", 
            "stations": [16],},    

    "france":{"coord":"france_gauges_france",
            "fn":"france_qobs_daily", 
            "stations": [12220010],},    

    "hygon":{"coord":"hygon_gauges_hygon",
            "fn":"hygon_qobs_daily", 
            "stations": [91000001, 15300002],},

                     }

#%% read model 

# root = r"p:\11208719-interreg\wflow\o_rwsinfo"
root = r"p:\11208719-interreg\wflow\p_geulrur"
# config_fn = "run_rwsinfo_eobs25.toml"
config_fn = "run_geulrur_eobs25.toml"
yml = r"p:\11208719-interreg\data\data_meuse.yml"
mod = WflowModel(root = root, config_fn=config_fn, data_libs=["deltares_data", yml], mode = "r")


#%% modeled data 

Folder_p = r"p:\11208719-interreg\wflow"

#routing runs
model_runs = {

    # "mod": {"case": "o_rwsinfo",
    #          "folder": "run_rwsinfo_eobs25"},

    "mod": {"case": "p_geulrur",
             "folder": "run_geulrur_eobs25"},
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

# caserun = "historic_daily_rwsinfo2"   
caserun = "historic_daily_geulrur"   
Folder_plots = r"d:\interreg\Plots" + "\\" + f"{caserun}"

if not os.path.exists(Folder_plots):
    os.mkdir(Folder_plots)

#%% make dataset 

var = "Q"

start = '1980-01-01 01:00:00'
end = '2021-12-31 23:00:00'
rng = pd.date_range(start, end, freq="D")

for source in source_dic:
    print(source)
    coord_name = source_dic[source]["coord"] #coordinate in output_scalar_nc
    fn = source_dic[source]["fn"] #name of entry in datacatalog
    stations_sel = source_dic[source]["stations"] #selected stations 

    #get observed ds
    ds_obs = mod.data_catalog.get_geodataset(fn, variables = ["Q"], time_tuple=(start, end))
    ds_obs_sel = ds_obs.sel(wflow_id = stations_sel)
    #add coordinate run for plots laurene 
    ds_obs_sel = ds_obs_sel.assign_coords({"runs":"Obs."}).expand_dims("runs")
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
        
        #NB: in toml eobs25 start hour is 01 -- so in the resulting netcdf it is also 1 hour
        #correct it here now, but better to rerun model and correct toml start time!
        ds_mod_sel = ds_mod_sel.resample(time="D").mean("time")
        

    #combine obs and runs in one dataset 
    ds = xr.concat([ds_obs_sel, ds_mod_sel], dim = "runs").to_dataset()
    ds = ds.load()


    #make plots
    start_long = '1981-01-01'
    end_long =  '2021-12-31'
    start_1 =  '2010-11-01'
    end_1 = '2011-03-01'
    start_2 =  '2011-03-01'
    end_2 =  '2011-10-31'
    start_3 =  '2015-01-01'
    end_3 = '2015-12-31'

    #empty df for performance
    df_perf_source = pd.DataFrame()
    for station_id in ds.stations.values:
        print(station_id)
        try: 
            station_name = ds["station_name"].sel(stations=station_id).values
        except:
            station_name = str(station_id)
        if len(ds["Q"].sel(runs = "Obs.", stations = station_id)) > 0:    
            runs_sel = list(runs_dict.keys()) 
            plot_colors = colors[:len(runs_dict)]
            dsq = ds.sel(stations = station_id).sel(time = slice('1981-01-01', None), runs = runs_sel + ["Obs."])#.dropna(dim='time')
            #plot hydro
            plot_hydro(dsq, start_long, end_long, start_1, end_1, start_2, end_2, start_3, end_3, runs_sel, plot_colors, Folder_plots, f"{source}_{station_name}_{station_id}" , save=True)
            plt.close()
            
            #make plot using function
            #dropna for signature calculations. 
            #start later for for warming up
            dsq = ds['Q'].sel(stations = station_id, runs = runs_sel + ["Obs."]).sel(time = slice('1981-01-01', '2021-12-31')).to_dataset().dropna(dim='time')
            if len(dsq.time)>366*2:
                dsq_perf = plot_signatures(dsq, runs_sel, plot_colors, Folder_plots, f"{source}_{station_name}_{station_id}" , save=True, window=7)
                dsq_perf_station = dsq_perf["performance"].sel(runs = key).to_dataframe()
                df_perf_source = pd.concat([df_perf_source, dsq_perf_station])
                plt.close()
            
    df_perf_source.to_csv(os.path.join(Folder_plots, f"performance_{source}.csv"))



#%% nse scores in report

metrics_sel = ["NSE", "KGE", "NSElog", "NM7Q", "MAXQ"]

df_score = pd.DataFrame()

for source in source_dic:
    df_perf_source = pd.read_csv(os.path.join(Folder_plots, f"performance_{source}.csv"))
    df_perf_source.index = df_perf_source["metrics"]

    for station_name in np.unique(df_perf_source.station_name):
        print(station_name)

        df_score_st = np.round(df_perf_source[df_perf_source.station_name == station_name].loc[metrics_sel][["performance"]],2).transpose().rename({"performance": station_name})
        df_score = pd.concat([df_score, df_score_st])

df_score.to_csv(os.path.join(Folder_plots, "performance_summary.csv"))