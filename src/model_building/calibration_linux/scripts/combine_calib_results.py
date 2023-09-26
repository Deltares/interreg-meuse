# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:30:44 2022

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
import glob 
from hydromt_wflow import WflowModel
import geopandas as gpd
from hydromt.stats import skills
import dask.array as daskarray

def intersection(a, b):
    return list(set(a).intersection(b))

#%%

import os, psutil, numpy as np # psutil may need to be installed
def usage():
    process = psutil.Process(os.getpid())
    return process.memory_info()[0] / float(2 ** 20)

def memory_usage_psutil():
    # return the memory usage in percentage like top
    process = psutil.Process(os.getpid())
    mem = process.memory_percent()
    return mem

usage() # initial memory usage
memory_usage_psutil() # memory usage percenatge

#%% read model and observations 
fs = 8

#windows path
Folder_plots = r"p:\11208719-interreg\wflow\j_waterschaplimburg\runs_calibration_linux_01\Results\Plots_fr"
root = r"p:\11208719-interreg\wflow\j_waterschaplimburg\runs_calibration_linux_01"
config_folder = r"n:\My Documents\unix-h6\interreg-meuse\src\model_building\calibration_linux\config"
obs_catalog = os.path.join(config_folder,"spw_windows.yml")

#linux path
Folder_plots = r"/p/11208719-interreg/wflow/j_waterschaplimburg/runs_calibration_linux_01/Results/Plots_fr"
root = r"/p/11208719-interreg/wflow/j_waterschaplimburg/runs_calibration_linux_01"
config_folder = r"/u/bouaziz/interreg-meuse/src/model_building/calibration_linux/config"
obs_catalog = os.path.join(config_folder,"spw.yml")

#%%
#calibration start end time 
start = '2006-01-01 01:00:00' #with warm up (technically... but memory errors)
end = '2011-12-31 00:00:00'

start_all = '2006-01-01 01:00:00' #without warm up 
end_all = '2011-12-31 00:00:00'

start_cal = '2006-01-01 01:00:00'
end_cal = '2011-12-31 00:00:00'

start_eval = '2006-01-01 01:00:00'
end_eval = '2011-12-31 00:00:00'

start_hyd = "2006-09-01"
end_hyd = "2011-08-31"

plot_year_hydro = 2008

timestep = 3600
if timestep == 3600:
    window = 7*24
    min_timesteps = 2 * 366 * 24
    freq = "H"
else: #assume daily 
    window = 7
    min_timesteps = 2 * 366 
    freq="D"

# selection_criteria = "dist_nse_nselog_nsenm7q_mm"
# selection_criteria = "dist_nse_nselog_nsenm7q_mm_maxq" #having maxq in the selection criteria leads to less good choices for monthly mean and cumulative - better to take it out!
selection_criteria = "dist_nse_nselog_nsenm7q_mm_cum" #add cum instead of maxq as criteria!

#toml file
toml_default_fn = "wflow_sbm_calibration.toml" 

source_data = "hydroportail" #or spw

if source_data == "spw":
    #observed data spw
    obs_ts_fn_daily_sel = "meuse-hydro_timeseries_spw_selection100km2_Q" #this is daily!
    obs_ts_fn = "meuse-hydro_timeseries_spw_hourly"
    prefix = "Q_spw" #prefix/header in csv file 
    obs_fn = "spw"
    index_name = "id"
else:
    #french data hourly
    obs_ts_fn = "meuse-hydro_timeseries_hydroportail_hourly"
    prefix = "Q_rws" #prefix/header in csv file 
    obs_fn = "Sall"
    index_name = "wflow_id"

#suffix observed data in staticmaps

    

def get_plotting_position(dsq_max, ascending=True):
    a=0.3
    b = 1.-2.*a
    max_y = np.round(dsq_max.max().values)
    ymin, ymax = 0, max_y
    p1 = ((np.arange(1,len(dsq_max.time)+1.)-a))/(len(dsq_max.time)+b)
    RP1 = 1/(1-p1)
    gumbel_p1 = -np.log(-np.log(1.-1./RP1))
    plotting_positions = dsq_max.sortby(dsq_max, ascending=ascending)
    return gumbel_p1, plotting_positions, max_y


# print("Make a list of output files and of parameter values")
filenames = glob.glob(os.path.join(root, "output_*.csv")) 
# psets = [filename.split("\\")[-1].split(".csv")[0].split("output_")[1] for filename in filenames]
# #make sure param name does not contain _
# psets_renamed = [filename.replace("floodplain_volume", "floodplainvolume") for filename in psets]
# psets_renamed = [filename.replace("storage_wood", "storagewood") for filename in psets]

# params_list = []
# for param in psets_renamed[0].split("_"):
#     param_name = param.split("~")[0]
#     params_list.append(param_name)

# param_values = dict()
# for param_name in params_list:
#     pval = [filename.split(f"{param_name}~")[1].split("_")[0] for filename in psets_renamed]
#     param_values[param_name] = {"param_values":pval}


print("Reading wflow model")
mod = WflowModel(root, config_fn=toml_default_fn, data_libs = obs_catalog, mode="r")

print("Reading observations")
obs = mod.data_catalog.get_geodataset(obs_ts_fn, geom=mod.basins)
obs = obs.vector.to_crs(mod.crs)
obs = obs.compute()
if source_data == "spw":
    obs = obs.assign_coords({
                            # "id_spw" : ("id", list(obs["id_spw"].values)),
                            "Nom" : ("index", list(obs["Nom"].values)),
                            "Riviere" : ("index", list(obs["Riviere"].values)),
                            "Bassin Versant" : ("index", list(obs["Bassin Versant"].values)),
                            "Superficie BV km2" : ("index", list(obs["Superficie BV km2"].values)),
                                            })
obs = obs["Q"]


print("Reading gauges obs fn")
gdf_gauges = mod.staticgeoms[f"gauges_{obs_fn}"]
gdf_gauges = gdf_gauges.set_index(f"{index_name}")
xs, ys = np.vectorize(lambda p: (p.xy[0][0], p.xy[1][0]))(gdf_gauges["geometry"])
idxs_gauges = mod.staticmaps.raster.xy_to_idx(xs, ys)
if source_data == "spw":
    #for spw stations only
    #select only a selection of stations instead of all stations to reduce run times using pre selection made for spw
    obs_sel = mod.data_catalog.get_geodataset(obs_ts_fn_daily_sel, geom=mod.basins)
    stations = gdf_gauges.index.values
    stations_sel = intersection(obs_sel.index.values, gdf_gauges.index.values)
else:
    stations = intersection(obs.index.values, gdf_gauges.index.values) 
    stations_sel = intersection(obs.index.values, gdf_gauges.index.values) 


#   kwargs:
#     drop_variables: ['Superficie BV km2'] ?? weird... 

#select a subset of stations to focus on for calibration 
# cal_stations = gpd.read_file(r"d:\SPW\Data\stations_Q_selection_calage.geojson")
# cal_stations.index = cal_stations.id

#%% fill dataset 

# #make dataset
# print("make dataset with output")
# variables = ['Q']    
    
# #calibration start and end time
# rng = pd.date_range(start, end, freq=freq)

# # S = np.zeros((len(rng), len(stations), len(psets)))
# # v = (('time', 'index', 'pset'), S)
# # h = {k:v for k in variables}

# # ds = xr.Dataset(
# #         data_vars=h, 
# #         coords={'time': rng,
# #                 'index': stations,
# #                 'pset': psets})
# # ds = ds * np.nan

# # https://stackoverflow.com/questions/73769633/initialize-larger-than-memory-xarray-dataset
# zeros = daskarray.zeros((len(rng), len(stations), len(psets)), chunks = (len(rng), 1, len(psets)), dtype='float64') #'uint8'

# ds = xr.Dataset(
#     data_vars = dict(
#     #discharge
#     Q=(["time", "index", "pset"], zeros),
#     ),

#     coords = dict(
#     index = stations,
#     pset = psets,
#     time = rng,
#     )
# )
# ds = ds * np.nan


# kwargs = {"index_col":0, "parse_dates":True, "header":0}

# print("fill dataset with output")
# for i, filename in enumerate(filenames):
#     print(i)
#     pset = filename.split("\\")[-1].split(".csv")[0].split("output_")[1]
#     data = pd.read_csv(filename, **kwargs)
#     for station in stations:
#         ds["Q"].loc[dict(pset = pset, index=station)] = data[f"{prefix}_{station}"].loc[start:end]

# #read default run before calibration 
# # run_ref = pd.read_csv(r"p:\11208186-spw\Models\Wflow\wflow_wallonie_rivers_gauges_global_fldplain_rz100km2_aardewerk1_lu_riv10\run_default\output.csv", **kwargs)
# # ds_ref = run_ref.to_xarray()

# # add params values
# print("add param values as coords")
# for param in param_values:
#     pval = param_values[param]["param_values"]
#     pval_float = [float(i) for i in pval]
#     # print(param, pval_float)
#     ds = ds.assign_coords({param : ("pset", pval_float)})

# #add empty name coordinate
# list_dummy_names = ["name xxxxxxxxxxx @ river xxxxxxxxxxx"] * len(ds.index)
# ds = ds.assign_coords({"name" : ("index", list_dummy_names)})

#%% same but separate nc saving for each param set 

rng = pd.date_range(start, end, freq=freq)

kwargs = {"index_col":0, "parse_dates":True, "header":0}

print("fill dataset with output")
for i, filename in enumerate(filenames):
    filename_fn_out = os.path.join(Folder_plots, f"ds_output_pset{i:04d}.nc")
    if not os.path.exists(filename_fn_out):
        print(i)
        pset = filename.split("\\")[-1].split(".csv")[0].split("output_")[1]

        # zeros = daskarray.zeros((len(rng), len(stations), len(pset)), chunks = (len(rng), 1, len(pset)), dtype='float64') #'uint8'
        zeros = np.zeros((len(rng), len(stations), len([pset])), dtype='float64') 
        ds = xr.Dataset(
            data_vars = dict(
            #discharge
            Q=(["time", "index", "pset"], zeros),
            ),

            coords = dict(
            index = stations,
            pset = [pset],
            time = rng,
            )
        )
        ds = ds * np.nan

        data = pd.read_csv(filename, **kwargs)
        for station in stations:
            ds["Q"].loc[dict(pset = pset, index=station)] = data[f"{prefix}_{station}"].loc[start:end]
        
        ds.to_netcdf(filename_fn_out)
        
        #free up memory?
        ds.close()
        del data
        del ds 

        print(memory_usage_psutil())

#%% reopen separate files:

# ds = xr.open_mfdataset(os.path.join(Folder_plots, "ds_output_pset*.nc"))#.load()
ds = xr.open_mfdataset(os.path.join(Folder_plots, "ds_output_pset*.nc"), chunks={"index":1})
#only select a subset
ds = ds.sel(index = stations_sel)

# add params values
print("Make a list of output files and of parameter values")
psets = list(ds.pset.values)
#make sure param name does not contain _
psets_renamed = [filename.replace("floodplain_volume", "floodplainvolume") for filename in psets]
psets_renamed = [filename.replace("storage_wood", "storagewood") for filename in psets]

params_list = []
for param in psets_renamed[0].split("_"):
    param_name = param.split("~")[0]
    params_list.append(param_name)

param_values = dict()
for param_name in params_list:
    pval = [filename.split(f"{param_name}~")[1].split("_")[0] for filename in psets_renamed]
    param_values[param_name] = {"param_values":pval}


print("add param values as coords")
for param in param_values:
    pval = param_values[param]["param_values"]
    pval_float = [float(i) for i in pval]
    # print(param, pval_float)
    ds = ds.assign_coords({param : ("pset", pval_float)})

#add empty name coordinate
list_dummy_names = ["name xxxxxxxxxxx @ river xxxxxxxxxxx"] * len(ds.index)
ds = ds.assign_coords({"name" : ("index", list_dummy_names)})


#%% calc perf indicators 

# #add a coord. 
# # ds["period"] = ["all","cal", "eval"]
# ds["period"] = ["all"]

# #add variables. 
# for perf_ind in perf_inds:
#     ds[perf_ind] = (('index', 'pset', 'period'), np.zeros((len(ds["index"]), len(ds['pset']), len(ds['period'])), dtype=np.float32)*np.nan)

# for period in all: #ds.period:
#     if period == "cal":
#         start = start_cal
#         end = end_cal
#     elif period == "eval":
#         start = start_eval
#         end = end_eval
#     else: #full period
#         start = start_all
#         end = end_all

print("Analysing results at the different stations")
periods = ["all"]
ds["period"] = periods
perf_inds = ["nse", "nse_log", "kge", "nse_nm7q", "nse_maxq", "nse_mm", "nse_cum", "dist_nse_nselog", "dist_nse_nselog_nsenm7q", "dist_nse_nselog_nsenm7q_mm", "dist_nse_nselog_nsenm7q_mm_maxq", "dist_nse_nselog_nsenm7q_mm_maxq_cum", "dist_nse_nselog_nsenm7q_mm_cum"]

for i in range(len(gdf_gauges.index)):
    st = gdf_gauges.index.values[i]
    if (st in obs.index) & (st in ds.index):
        print(f"Station {st} ({i+1}/{len(gdf_gauges.index)})")

        #filename output
        filename_fn_st = os.path.join(Folder_plots, f"ds_output_st{st}.nc")

        if not os.path.exists(filename_fn_st):

            #name of discharge in obs netcdf and in csv !
            discharge_name = "Q"

            #load sim data for st 
            print("loading the data")
            ds_st = ds.sel(index = st).load()
            ds_st["period"] = periods
            #add variables. 
            
            for perf_ind in perf_inds:
                ds_st[perf_ind] = (('pset', 'period'), np.zeros((len(ds_st['pset']), len(ds_st['period'])), dtype=np.float32)*np.nan)

            for period in periods: #ds.period:
                if period == "cal":
                    start = start_cal
                    end = end_cal
                elif period == "eval":
                    start = start_eval
                    end = end_eval
                else: #full period
                    start = start_all
                    end = end_all


                #check if there is observed data
                # Read observation at the station
                print("calculating perf ind")
                obs_i = obs.sel(index=st).sel(time=slice(start, end))
                mask = ~obs_i.isnull().compute()
                try:
                    obs_i_nan = obs_i.where(mask, drop = True)
                except:
                    obs_i_nan = obs_i[discharge_name].where(mask[discharge_name], drop = True)
                
                if len(obs_i_nan.time) > min_timesteps: # should be changed if hourly! make sure enough observed data length   
                    #nm7q
                    obs_i_nan_nm7q = obs_i_nan.rolling(time = window).mean().resample(time = 'A').min('time').compute()
                    #mean monthly
                    obs_i_nan_mm = obs_i_nan.groupby("time.month").mean().compute()
                    #max annual 
                    obs_i_nan_max = obs_i_nan.sel(time=slice(start_hyd, end_hyd)).resample(time = 'AS-Sep').max('time').compute()
                    #cum
                    obs_i_nan_cum = obs_i_nan.cumsum("time").compute()

                    #name 
                    if source_data == "spw":
                        name = str(obs_i["Nom"].values) + " @ " + str(obs_i["Riviere"].values)
                    else:
                        name = str(obs_i["Libellé"].values) 
                    ds["name"].loc[dict(index=st)] = name

                    #Read simulations
                    for pset in ds_st.pset.values:
                        # sim_i_nan = ds[discharge_name].sel(index = st, time=obs_i_nan.time, pset=pset).compute()
                        sim_i_nan = ds_st[discharge_name].sel(time=obs_i_nan.time, pset=pset).compute()
                        #nm7q
                        sim_i_nan_nm7q = sim_i_nan.rolling(time = window).mean().resample(time = 'A').min('time').compute()
                        #mean monthly
                        sim_i_nan_mm = sim_i_nan.groupby("time.month").mean().compute()
                        #max annual 
                        sim_i_nan_max = sim_i_nan.sel(time=slice(start_hyd, end_hyd)).resample(time = 'AS-Sep').max('time').compute()
                        #cum
                        sim_i_nan_cum = sim_i_nan.cumsum('time').compute()

                        if len(obs_i_nan.time) > min_timesteps: # should be changed if hourly! make sure enough observed data length   
                        # plot the different runs and compute performance
                            nse = skills.nashsutcliffe(sim_i_nan, obs_i_nan).values.round(4)
                            nse_log = skills.lognashsutcliffe(sim_i_nan, obs_i_nan).values.round(4)
                            kge = skills.kge(sim_i_nan, obs_i_nan)['kge'].values.round(4)
                            nse_nm7q = skills.nashsutcliffe(sim_i_nan_nm7q, obs_i_nan_nm7q).values.round(4)
                            nse_maxq = skills.nashsutcliffe(sim_i_nan_max, obs_i_nan_max).values.round(4)
                            nse_mm = skills.nashsutcliffe(sim_i_nan_mm, obs_i_nan_mm, dim = "month").values.round(4)
                            nse_cum = skills.nashsutcliffe(sim_i_nan_cum, obs_i_nan_cum).values.round(4)

                            # dist_nse_nselog = np.sqrt((1-nse)**2 + (1-nse_log)**2)
                            # dist_nse_nselog_nsenm7q = np.sqrt((1-nse)**2 + (1-nse_log)**2 + (1-nse_nm7q)**2)
                            # dist_nse_nselog_nsenm7q_mm = np.sqrt((1-nse)**2 + (1-nse_log)**2 + (1-nse_nm7q)**2 + (1-nse_mm)**2)
                            # dist_nse_nselog_nsenm7q_mm_maxq = np.sqrt((1-nse)**2 + (1-nse_log)**2 + (1-nse_nm7q)**2 + (1-nse_mm)**2 + (1-nse_maxq)**2)
                            # dist_nse_nselog_nsenm7q_mm_cum = np.sqrt((1-nse)**2 + (1-nse_log)**2 + (1-nse_nm7q)**2 + (1-nse_mm)**2 + (1-nse_cum)**2)
                            # dist_nse_nselog_nsenm7q_mm_maxq_cum = np.sqrt((1-nse)**2 + (1-nse_log)**2 + (1-nse_nm7q)**2 + (1-nse_mm)**2 + (1-nse_maxq)**2 + (1-nse_cum)**2)
                            
                            #fill in dataset ds_st!
                            ds_st["nse"].loc[dict(period = period, pset=pset)] = nse
                            ds_st["nse_log"].loc[dict(period = period, pset=pset)] = nse_log
                            ds_st["kge"].loc[dict(period = period, pset=pset)] = kge
                            ds_st["nse_nm7q"].loc[dict(period = period, pset=pset)] = nse_nm7q
                            ds_st["nse_maxq"].loc[dict(period = period, pset=pset)] = nse_maxq
                            ds_st["nse_mm"].loc[dict(period = period, pset=pset)] = nse_mm
                            ds_st["nse_cum"].loc[dict(period = period, pset=pset)] = nse_cum
                            
                            # ds_st["dist_nse_nselog"].loc[dict(period = period, pset=pset)] = dist_nse_nselog
                            # ds_st["dist_nse_nselog_nsenm7q"].loc[dict(period = period, pset=pset)] = dist_nse_nselog_nsenm7q
                            # ds_st["dist_nse_nselog_nsenm7q_mm"].loc[dict(period = period, pset=pset)] = dist_nse_nselog_nsenm7q_mm
                            # ds_st["dist_nse_nselog_nsenm7q_mm_maxq"].loc[dict(period = period, pset=pset)] = dist_nse_nselog_nsenm7q_mm_maxq
                            # ds_st["dist_nse_nselog_nsenm7q_mm_cum"].loc[dict(period = period, pset=pset)] = dist_nse_nselog_nsenm7q_mm_cum
                            # ds_st["dist_nse_nselog_nsenm7q_mm_maxq_cum"].loc[dict(period = period, pset=pset)] = dist_nse_nselog_nsenm7q_mm_maxq_cum

                    ds_st["dist_nse_nselog"] =  np.sqrt((1-ds_st["nse"])**2 + (1-ds_st["nse_log"])**2 )
                    ds_st["dist_nse_nselog_nsenm7q"] =  np.sqrt((1-ds_st["nse"])**2 + (1-ds_st["nse_log"])**2 + (1-ds_st["nse_nm7q"])**2)
                    ds_st["dist_nse_nselog_nsenm7q_mm"] =  np.sqrt((1-ds_st["nse"])**2 + (1-ds_st["nse_log"])**2 + (1-ds_st["nse_nm7q"])**2 + (1-ds_st["nse_mm"])**2 )
                    ds_st["dist_nse_nselog_nsenm7q_mm_maxq"] =  np.sqrt((1-ds_st["nse"])**2 + (1-ds_st["nse_log"])**2 + (1-ds_st["nse_nm7q"])**2 + (1-ds_st["nse_mm"])**2 + (1-ds_st["nse_maxq"])**2)
                    ds_st["dist_nse_nselog_nsenm7q_mm_cum"] =  np.sqrt((1-ds_st["nse"])**2 + (1-ds_st["nse_log"])**2 + (1-ds_st["nse_nm7q"])**2 + (1-ds_st["nse_mm"])**2 + (1-ds_st["nse_cum"])**2)
                    ds_st["dist_nse_nselog_nsenm7q_mm_maxq_cum"] =  np.sqrt((1-ds_st["nse"])**2 + (1-ds_st["nse_log"])**2 + (1-ds_st["nse_nm7q"])**2 + (1-ds_st["nse_mm"])**2 + (1-ds_st["nse_maxq"])**2 + (1-ds_st["nse_cum"])**2)


                            # #fill in dataset ds
                            # ds["nse"].loc[dict(period = period, pset=pset, index=st)] = nse
                            # ds["nse_log"].loc[dict(period = period, pset=pset, index=st)] = nse_log
                            # ds["kge"].loc[dict(period = period, pset=pset, index=st)] = kge
                            # ds["nse_nm7q"].loc[dict(period = period, pset=pset, index=st)] = nse_nm7q
                            # ds["nse_maxq"].loc[dict(period = period, pset=pset, index=st)] = nse_maxq
                            # ds["nse_mm"].loc[dict(period = period, pset=pset, index=st)] = nse_mm
                            # ds["dist_nse_nselog"].loc[dict(period = period, pset=pset, index=st)] = dist_nse_nselog
                            # ds["dist_nse_nselog_nsenm7q"].loc[dict(period = period, pset=pset, index=st)] = dist_nse_nselog_nsenm7q
                            # ds["dist_nse_nselog_nsenm7q_mm"].loc[dict(period = period, pset=pset, index=st)] = dist_nse_nselog_nsenm7q_mm
                            # ds["dist_nse_nselog_nsenm7q_mm_maxq"].loc[dict(period = period, pset=pset, index=st)] = dist_nse_nselog_nsenm7q_mm_maxq

            encoding = {
                    v: {"zlib": True, "dtype": "float32", } #"chunksizes": chunksizes
                    for v in ds_st.data_vars.keys()
                }
            print("write netcdf")
            ds_st.to_netcdf(filename_fn_st)
            

#%%
#save ds to netcdf -- would be too large to fit everything in 1 nc for the hourly runs. 
# print("write to netcdf")
# chunksizes = (1, ds.pset.size, ds.time.size)
# encoding = {
#         v: {"zlib": True, "dtype": "float32", "chunksizes": chunksizes}
#         for v in ds.data_vars.keys()
#     }
# # encoding["time"] = {"_FillValue": None}
# #todo: remove Q from ds because already saved in separate nc files. 
# ds.to_netcdf(os.path.join(Folder_plots, "ds_output.nc"))


#%% get best parameter set per station and associated param values 

if not os.path.exists(os.path.join(Folder_plots, "ds_output_best.nc")):
    print("create ds_output_best")

    ds_best = xr.Dataset(
        data_vars = dict(
        nse=(["index", "period"], np.zeros((len(ds.index), len(ds.period)))),
        nse_log=(["index", "period"], np.zeros((len(ds.index), len(ds.period)))),
        kge=(["index", "period"], np.zeros((len(ds.index), len(ds.period)))),
        nse_nm7q=(["index", "period"], np.zeros((len(ds.index), len(ds.period)))),
        nse_maxq=(["index", "period"], np.zeros((len(ds.index), len(ds.period)))),
        nse_mm=(["index", "period"], np.zeros((len(ds.index), len(ds.period)))),
        nse_cum=(["index", "period"], np.zeros((len(ds.index), len(ds.period)))),
        dist_nse_nselog=(["index", "period"], np.zeros((len(ds.index), len(ds.period)))),
        dist_nse_nselog_nsenm7q=(["index", "period"], np.zeros((len(ds.index), len(ds.period)))),
        dist_nse_nselog_nsenm7q_mm=(["index", "period"], np.zeros((len(ds.index), len(ds.period)))),
        dist_nse_nselog_nsenm7q_mm_cum=(["index", "period"], np.zeros((len(ds.index), len(ds.period)))),
        dist_nse_nselog_nsenm7q_mm_maxq=(["index", "period"], np.zeros((len(ds.index), len(ds.period)))),
        dist_nse_nselog_nsenm7q_mm_maxq_cum=(["index", "period"], np.zeros((len(ds.index), len(ds.period)))),
        #discharge
        Q=(["index", "time", "period"], np.zeros((len(ds.index), len(ds.time), len(ds.period)))),
        Qobs=(["index", "time"], np.zeros((len(ds.index), len(ds.time)))),
        ),

        coords = dict(
        index = ds.index,
        name = ds.name,
        period = ds.period,
        time = ds.time,
        )

    )

    #add params
    for param in param_values:
        ds_best = ds_best.assign({param : (("index","period"), np.zeros((len(ds.index), len(ds.period))) )})

    ds_best = ds_best * np.nan


    print("fill nc with best data")
    for period in ds.period:
        for st in ds.index.values:
            print(st)
            #open previously saved ds_st file 
            ds_st = xr.open_dataset(os.path.join(Folder_plots, f"ds_output_st{st}.nc"))
            # if not np.isnan(ds[selection_criteria].sel(index=st, period = period).min().values):        
            #     pset_best = ds[selection_criteria].sel(index=st, period = period).idxmin().values
            #     ds_best_st = ds.sel(pset = pset_best, index=st, period=period)
            if not np.isnan(ds_st[selection_criteria].sel(period = period).min().values):        
                pset_best = ds_st[selection_criteria].sel(period = period).idxmin().values
                ds_best_st = ds_st.sel(pset = pset_best, period=period)
                #fill in ds_best
                for perf_ind in perf_inds:
                    ds_best[perf_ind].loc[dict(period=period, index=st)] = ds_best_st[perf_ind]
                for param in params_list:
                    ds_best[param].loc[dict(period=period, index=st)] = ds_best_st[param]
                #fill in Q
                ds_best["Q"].loc[dict(index=st, period = period)] = ds_best_st["Q"]
                #Qobs
                if st in obs.index.values:
                    ds_best["Qobs"].loc[dict(index=st)] = obs.sel(index=st, time = ds_best.time)
                

    # ds_best.loc[dict(period=period, index=st)]

    #save ds to netcdf
    print("write to netcdf")
    ds_best.to_netcdf(os.path.join(Folder_plots, "ds_output_best.nc"))
else:
    ds_best = xr.open_dataset(os.path.join(Folder_plots, "ds_output_best.nc"))


#%% plots signature all and best, cumulative, monthly regime, low flows, high flows

# start = "2011-01-01"
# end = "2021-12-31"

# start_hyd = "2011-09-01"
# end_hyd = "2021-08-31"

period = "all"

print("make signature plots")
for st in ds.index.values:
    #open ds_st 
    ds_st = xr.open_dataset(os.path.join(Folder_plots, f"ds_output_st{st}.nc"), chunks = {"pset":100})
    # if (st in obs.index) & (not np.isnan(ds[selection_criteria].sel(index=st, period = period).min().values)):
    if (st in obs.index) & (not np.isnan(ds_st[selection_criteria].sel(period = period).min().values)):
        if not os.path.exists(os.path.join(Folder_plots, f"signatures_{ds_best.sel(index=st).name.values}_{st}.png")):
            print(st)
            fig, axes = plt.subplots(2,2, figsize=(12/2.54, 12/2.54))
            ax = axes.flatten()
            #mean monthly flow
            overlap_time = ds_best["Qobs"].sel(index=st, time = slice(start, end)).dropna("time").time.values
            # ds["Q"].sel(index=st, time = overlap_time).groupby("time.month").mean().plot(ax=ax[0], hue = "pset", add_legend=False, color = "grey")
            ds_st["Q"].sel(time = overlap_time).groupby("time.month").mean().plot(ax=ax[0], hue = "pset", add_legend=False, color = "grey")
            ds_best["Q"].sel(index=st, time = overlap_time, period = period).groupby("time.month").mean().plot(ax=ax[0], add_legend=False, color = "orange")
            ds_best["Qobs"].sel(index=st, time = overlap_time).groupby("time.month").mean().plot(ax=ax[0], add_legend=False, color = "k", linestyle = "--")
            ax[0].set_xticks(np.arange(1,13))
            ax[0].set_xlabel("")
            ax[0].set_ylabel("Q (m$^{3}$ s$^{1}$)", fontsize=fs)

            #cumulative 
            # ds["Q"].sel(index=st, time = overlap_time).cumsum("time").plot(ax=ax[1], hue = "pset", add_legend=False, color = "grey")
            ds_st["Q"].sel(time = overlap_time).cumsum("time").plot(ax=ax[1], hue = "pset", add_legend=False, color = "grey")
            ds_best["Q"].sel(index=st, time = overlap_time, period = period).cumsum().plot(ax=ax[1], add_legend=False, color = "orange")
            ds_best["Qobs"].sel(index=st, time = overlap_time).cumsum().plot(ax=ax[1], add_legend=False, color = "k", linestyle = "--")
            ax[1].set_ylabel("Qcum (m$^{3}$ s$^{1}$)", fontsize=fs)

            #lowest flows
            obs_i_nan_nm7q = ds_best["Qobs"].sel(index=st, time = overlap_time).rolling(time = window).mean().resample(time = 'A').min('time').compute()
            best_i_nan_nm7q = ds_best["Q"].sel(index=st, time = overlap_time, period=period).rolling(time = window).mean().resample(time = 'A').min('time').compute()

            gumbel_p1, plotting_positions_obs, ymax_obs = get_plotting_position(obs_i_nan_nm7q, ascending=False)
            gumbel_p1, plotting_positions_best, ymax_best = get_plotting_position(best_i_nan_nm7q, ascending=False)
            ymax = np.maximum(ymax_obs, ymax_best) 

            ts = [2., 5.,10.,30.]

            #takes too long
            # for pset in ds.pset.values:
            #     # sim_i_nan = ds["Q"].sel(index = st, time=overlap_time, pset=pset)
            #     sim_i_nan = ds_st["Q"].sel(time=overlap_time, pset=pset)
            #     sim_i_nan_nm7q = sim_i_nan.rolling(time = window).mean().resample(time = 'A').min('time')
            #     gumbel_p1, plotting_positions_sim, ymax_sim = get_plotting_position(sim_i_nan_nm7q, ascending=False)
            #     ax[2].plot(gumbel_p1, plotting_positions_sim, marker = '.', color = 'grey', linestyle = 'None', markersize = 6)

            ax[2].plot(gumbel_p1, plotting_positions_best, marker = 'o', color = 'orange', linestyle = 'None', label = 'best', markersize = 6)
            ax[2].plot(gumbel_p1, plotting_positions_obs, marker = '+', color = 'k', linestyle = 'None', label = 'obs.', markersize = 6)
            ax[2].legend(fontsize=fs)
            ax[2].set_ylabel('NM7Q (m$^{3}$ s$^{1}$)', fontsize = fs)
            ax[2].set_xlabel('Plotting position and RP' , fontsize = fs)
            for t in ts:
                ax[2].axvline(-np.log(-np.log(1-1./t)),c='0.5', alpha=0.4)
                ax[2].text(-np.log(-np.log(1-1./t)),ymax*1.05,f'{t:.0f}y',  ha="center", va="bottom", fontsize=fs) #fontsize = fs,
            

            #highest flows starting in sept!!
            overlap_time_max = ds_best["Qobs"].sel(index=st, time = slice(start_hyd, end_hyd)).dropna("time").time.values
            
            obs_i_nan_max = ds_best["Qobs"].sel(index=st, time = overlap_time_max).resample(time = 'AS-Sep').max('time').compute()
            best_i_nan_max = ds_best["Q"].sel(index=st, time = overlap_time_max, period=period).resample(time = 'AS-Sep').max('time').compute()
            
            #plotting pos
            gumbel_p1, plotting_positions_obs, ymax_obs = get_plotting_position(obs_i_nan_max)
            gumbel_p1, plotting_positions_best, ymax_best = get_plotting_position(best_i_nan_max)
            ymax = np.maximum(ymax_obs, ymax_best)    

            #takes too long
            # for pset in ds.pset.values:
            #     # sim_i_nan = ds["Q"].sel(index = st, time=overlap_time_max, pset=pset)
            #     sim_i_nan = ds_st["Q"].sel(time=overlap_time_max, pset=pset)
            #     sim_i_nan_max = sim_i_nan.resample(time = 'AS-Sep').max('time')
            #     gumbel_p1, plotting_positions_sim, ymax_sim = get_plotting_position(sim_i_nan_max)
            #     ax[3].plot(gumbel_p1, plotting_positions_sim, marker = '.', color = 'grey', linestyle = 'None', markersize = 6)
            
            ax[3].plot(gumbel_p1, plotting_positions_best, marker = 'o', color = 'orange', linestyle = 'None', label = 'best', markersize = 6)
            ax[3].plot(gumbel_p1, plotting_positions_obs, marker = '+', color = 'k', linestyle = 'None', label = 'obs.', markersize = 6)
            ax[3].legend(fontsize=fs)
            ax[3].set_ylabel('max. annual Q (m$^{3}$ s$^{1}$)', fontsize = fs)
            ax[3].set_xlabel('Plotting position and RP' , fontsize = fs)
            for t in ts:
                ax[3].axvline(-np.log(-np.log(1-1./t)),c='0.5', alpha=0.4)
                ax[3].text(-np.log(-np.log(1-1./t)),ymax*1.05,f'{t:.0f}y',  ha="center", va="bottom", fontsize = fs)

            for axe in ax:
                axe.tick_params(axis="both", labelsize = fs)
                axe.set_title("")

            plt.suptitle(f"{ds_best.sel(index=st).name.values} ({st})", fontsize = fs)

            plt.tight_layout()
            plt.savefig(os.path.join(Folder_plots, f"signatures_{ds_best.sel(index=st).name.values}_{st}.png"), dpi=300)
            plt.close()
        ds_st.close()
        del ds_st






#%% plot  hydro
print("plot hydro")
for st in ds.index.values:
    ds_st = xr.open_dataset(os.path.join(Folder_plots, f"ds_output_st{st}.nc"))
    # if (st in obs.index) & (not np.isnan(ds[selection_criteria].sel(index=st, period = period).min().values)):
    if (st in obs.index) & (not np.isnan(ds_st[selection_criteria].sel(period = period).min().values)):
        print(st)
        fig, axes = plt.subplots(2,1, figsize=(12/2.54, 12/2.54),  sharey=True)
        ax = axes.flatten()
        # ds["Q"].sel(index=st, time = slice(start, end)).plot(ax=ax[0], hue = "pset", add_legend=False, color = "grey", linewidth=0.8)
        # ds_st["Q"].sel(time = slice(start, end)).plot(ax=ax[0], hue = "pset", add_legend=False, color = "grey", linewidth=0.8)
        ds_best["Q"].sel(index=st, time = slice(start, end), period = period).plot(ax=ax[0], add_legend=False, color = "orange", linewidth=0.8)
        ds_best["Qobs"].sel(index=st, time = slice(start, end)).plot(ax=ax[0], add_legend=False, color = "k", linestyle = "--", linewidth=0.8)
        # 2015
        year = plot_year_hydro
        # ds["Q"].sel(index=st, time = slice(f"{year}-01-01", f"{year}-12-31")).plot(ax=ax[1], hue = "pset", add_legend=False, color = "grey", alpha=0.5, linewidth=0.8)
        ds_st["Q"].sel(time = slice(f"{year}-01-01", f"{year}-12-31")).plot(ax=ax[1], hue = "pset", add_legend=False, color = "grey", alpha=0.5, linewidth=0.8)
        ds_best["Q"].sel(index=st, time = slice(f"{year}-01-01", f"{year}-12-31"), period = period).plot(ax=ax[1], add_legend=False, color = "orange", linewidth=0.8)
        ds_best["Qobs"].sel(index=st, time = slice(f"{year}-01-01", f"{year}-12-31")).plot(ax=ax[1], add_legend=False, color = "k", linestyle = "--", linewidth=0.8)

        for axe in ax:
            axe.tick_params(axis="both", labelsize=fs)
            axe.set_xlabel("")
            axe.set_ylabel("Q m$^{3}$ s$^{1}$", fontsize=fs)
        ax[0].set_title(f"{ds_best.sel(index=st).name.values} ({st})", fontsize=fs)
        ax[1].set_title("")
        plt.tight_layout()
        plt.savefig(os.path.join(Folder_plots, f"hydro_{ds_best.sel(index=st).name.values}_{st}.png"), dpi=300)
        plt.close()
    ds_st.close()
    del ds_st





#%% dotty plots 
print("plot dotty plots")
period = "all"
for st in ds.index.values:
    ds_st = xr.open_dataset(os.path.join(Folder_plots, f"ds_output_st{st}.nc"))
    # if (st in obs.index) & (not np.isnan(ds[selection_criteria].sel(index=st, period = period).min().values)):
    if (st in obs.index) & (not np.isnan(ds_st[selection_criteria].sel(period = period).min().values)):
        print(st)
        for perf_ind in perf_inds:
            fig, axes = plt.subplots(round(len(params_list)/2),2, figsize=(12/2.54, 12/2.54))
            ax = axes.flatten()
            for i, param in enumerate(params_list):
                # ax[i].plot(ds[param], ds[perf_ind].sel(period = "all", index = st), linestyle="None", marker=".", color = "grey")
                ax[i].plot(ds_st[param], ds_st[perf_ind].sel(period = "all"), linestyle="None", marker=".", color = "grey")
                ax[i].plot(ds_best[param].sel(index=st, period =period), ds_best[perf_ind].sel(period = "all", index = st), linestyle="None", marker=".", color = "orange")
                if not perf_ind.startswith("dist"):
                    ax[i].set_ylim([0,1])
                ax[i].set_xlabel(param, fontsize = fs)
                ax[i].tick_params(axis="both", labelsize=fs)
                ax[i].set_ylabel(f"{perf_ind}", fontsize = fs)
            plt.suptitle(f"{ds_best.sel(index=st).name.values} ({st})", fontsize = fs)
            plt.tight_layout()
            plt.savefig(os.path.join(Folder_plots, f"dottyplots_{ds_best.sel(index=st).name.values}_{st}_{perf_ind}.png"), dpi=300)
            plt.close()
    # ds_st.close()
    # del ds_st


