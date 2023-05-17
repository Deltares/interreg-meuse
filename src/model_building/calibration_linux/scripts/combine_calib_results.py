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
#%% read model and observations 
fs = 8

#windows path
# Folder_plots = r"p:\11208186-spw\Models\Wflow\wflow_wallonie_rivers_gauges_global_fldplain_rz100km2_aardewerk1_lu_riv10\runs_calibration_linux_01\Results\Plots"
# root = r"p:\11208186-spw\Models\Wflow\wflow_wallonie_rivers_gauges_global_fldplain_rz100km2_aardewerk1_lu_riv10\runs_calibration_linux_01"
# config_folder = r"p:\11208186-spw\src\calibration_linux\config"
# obs_catalog = os.path.join(config_folder,"spw_windows.yml")

#linux path
Folder_plots = r"/p/11208186-spw/Models/Wflow/wflow_wallonie_rivers_gauges_global_fldplain_rz100km2_aardewerk1_lu_riv10/runs_calibration_linux_01/Results/Plots"
root = r"/p/11208186-spw/Models/Wflow/wflow_wallonie_rivers_gauges_global_fldplain_rz100km2_aardewerk1_lu_riv10/runs_calibration_linux_01"
config_folder = r"/p/11208186-spw/src/calibration_linux/config"
obs_catalog = os.path.join(config_folder,"spw.yml")

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


print("Make a list of output files and of parameter values")
filenames = glob.glob(os.path.join(root, "output_k*.csv"))
psets = [filename.split("\\")[-1].split(".csv")[0].split("output_")[1] for filename in filenames]
#make sure param name does not contain _
psets_renamed = [filename.replace("floodplain_volume", "floodplainvolume") for filename in psets]

params_list = []
for param in psets_renamed[0].split("_"):
    param_name = param.split("~")[0]
    params_list.append(param_name)

param_values = dict()
for param_name in params_list:
    pval = [filename.split(f"{param_name}~")[1].split("_")[0] for filename in psets_renamed]
    param_values[param_name] = {"param_values":pval}


print("Reading wflow model")
toml_default_fn = "wflow_sbm_calibration.toml"
mod = WflowModel(root, config_fn=toml_default_fn, data_libs = obs_catalog, mode="r")


print("Reading gauges obs fn")
obs_fn = "stations_obs"
gdf_gauges = mod.staticgeoms[f"gauges_{obs_fn}"]
gdf_gauges = gdf_gauges.set_index("id")
xs, ys = np.vectorize(lambda p: (p.xy[0][0], p.xy[1][0]))(gdf_gauges["geometry"])
idxs_gauges = mod.staticmaps.raster.xy_to_idx(xs, ys)
stations = gdf_gauges.index.values


print("Reading observations")
obs_ts_fn = "meuse-hydro_timeseries_spw_selection100km2_Q"
obs = mod.data_catalog.get_geodataset(obs_ts_fn, geom=mod.basins)
obs = obs.vector.to_crs(mod.crs)
obs = obs.compute()
obs = obs["Q"]
#   kwargs:
#     drop_variables: ['Superficie BV km2'] ?? weird... 

#select a subset of stations to focus on for calibration 
# cal_stations = gpd.read_file(r"d:\SPW\Data\stations_Q_selection_calage.geojson")
# cal_stations.index = cal_stations.id

#%% fill dataset 

#make dataset
print("make dataset with output")
variables = ['Q']    
    
start = '2010-01-01'
end = '2021-12-31'
rng = pd.date_range(start, end)

S = np.zeros((len(rng), len(stations), len(psets)))
v = (('time', 'index', 'pset'), S)
h = {k:v for k in variables}

ds = xr.Dataset(
        data_vars=h, 
        coords={'time': rng,
                'index': stations,
                'pset': psets})
ds = ds * np.nan

kwargs = {"index_col":0, "parse_dates":True, "header":0}

print("fill dataset with output")
for i, filename in enumerate(filenames):
    print(i)
    pset = filename.split("\\")[-1].split(".csv")[0].split("output_")[1]
    data = pd.read_csv(filename, **kwargs)
    for station in stations:
        ds["Q"].loc[dict(pset = pset, index=station)] = data[f"Q_{station}"]

#read default run before calibration 
# run_ref = pd.read_csv(r"p:\11208186-spw\Models\Wflow\wflow_wallonie_rivers_gauges_global_fldplain_rz100km2_aardewerk1_lu_riv10\run_default\output.csv", **kwargs)
# ds_ref = run_ref.to_xarray()

# add params values
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

#add a coord. 
# ds["period"] = ["all","cal", "eval"]
ds["period"] = ["all"]

#add variables. 
perf_inds = ["nse", "nse_log", "kge", "nse_nm7q", "nse_mm", "dist_nse_nselog", "dist_nse_nselog_nsenm7q", "dist_nse_nselog_nsenm7q_mm"]
for perf_ind in perf_inds:
    ds[perf_ind] = (('index', 'pset', 'period'), np.zeros((len(ds["index"]), len(ds['pset']), len(ds['period'])), dtype=np.float32)*np.nan)

print("Analysing results at the different stations")
for period in ds.period:
    if period == "cal":
        start = "2011-01-01"
        end = "2016-12-31"
    elif period == "eval":
        start = "2017-01-01"
        end = "2021-12-31"
    else: #full period
        start = "2011-01-01"
        end = "2021-12-31"

    for i in range(len(gdf_gauges.index)):
        st = gdf_gauges.index.values[i]
        if st in obs.index:
            print(f"Station {st} ({i+1}/{len(gdf_gauges.index)})")
            #name of discharge in obs netcdf and in csv !
            discharge_name = "Q"
            #check if there is observed data
            # Read observation at the station
            obs_i = obs.sel(index=st).sel(time=slice(start, end))
            mask = ~obs_i.isnull().compute()
            try:
                obs_i_nan = obs_i.where(mask, drop = True)
            except:
                obs_i_nan = obs_i[discharge_name].where(mask[discharge_name], drop = True)
            
            if len(obs_i_nan.time) > 2*366: # should be changed if hourly! make sure enough observed data length   
                obs_i_nan_nm7q = obs_i_nan.rolling(time = 7).mean().resample(time = 'A').min('time').compute()
                #mean monthly
                obs_i_nan_mm = obs_i_nan.groupby("time.month").mean()

                #name 
                name = obs_i["Nom"].values + " @ " + obs_i["Riviere"].values
                ds["name"].loc[dict(index=st)] = name

                #Read simulations
                for pset in ds.pset.values:
                    sim_i_nan = ds[discharge_name].sel(index = st, time=obs_i_nan.time, pset=pset)
                    sim_i_nan_nm7q = sim_i_nan.rolling(time = 7).mean().resample(time = 'A').min('time')
                    #mean monthly
                    sim_i_nan_mm = sim_i_nan.groupby("time.month").mean()

                    if len(obs_i_nan.time) > 2*366: # should be changed if hourly! make sure enough observed data length   
                    # plot the different runs and compute performance
                        nse = skills.nashsutcliffe(sim_i_nan, obs_i_nan).values.round(4)
                        nse_log = skills.lognashsutcliffe(sim_i_nan, obs_i_nan).values.round(4)
                        kge = skills.kge(sim_i_nan, obs_i_nan)['kge'].values.round(4)
                        nse_nm7q = skills.nashsutcliffe(sim_i_nan_nm7q, obs_i_nan_nm7q).values.round(4)
                        nse_mm = skills.nashsutcliffe(sim_i_nan_mm, obs_i_nan_mm, dim = "month").values.round(4)
                        dist_nse_nselog = np.sqrt((1-nse)**2 + (1-nse_log)**2)
                        dist_nse_nselog_nsenm7q = np.sqrt((1-nse)**2 + (1-nse_log)**2 + (1-nse_nm7q)**2)
                        dist_nse_nselog_nsenm7q_mm = np.sqrt((1-nse)**2 + (1-nse_log)**2 + (1-nse_nm7q)**2 + (1-nse_mm)**2)

                        #fill in dataset
                        ds["nse"].loc[dict(period = period, pset=pset, index=st)] = nse
                        ds["nse_log"].loc[dict(period = period, pset=pset, index=st)] = nse_log
                        ds["kge"].loc[dict(period = period, pset=pset, index=st)] = kge
                        ds["nse_nm7q"].loc[dict(period = period, pset=pset, index=st)] = nse_nm7q
                        ds["nse_mm"].loc[dict(period = period, pset=pset, index=st)] = nse_mm
                        ds["dist_nse_nselog"].loc[dict(period = period, pset=pset, index=st)] = dist_nse_nselog
                        ds["dist_nse_nselog_nsenm7q"].loc[dict(period = period, pset=pset, index=st)] = dist_nse_nselog_nsenm7q
                        ds["dist_nse_nselog_nsenm7q_mm"].loc[dict(period = period, pset=pset, index=st)] = dist_nse_nselog_nsenm7q_mm

#save ds to netcdf
print("write to netcdf")
ds.to_netcdf(os.path.join(Folder_plots, "ds_output.nc"))

#%% get best parameter set per station and associated param values 
ds_best = xr.Dataset(
    data_vars = dict(
    nse=(["index", "period"], np.zeros((len(ds.index), len(ds.period)))),
    nse_log=(["index", "period"], np.zeros((len(ds.index), len(ds.period)))),
    kge=(["index", "period"], np.zeros((len(ds.index), len(ds.period)))),
    nse_nm7q=(["index", "period"], np.zeros((len(ds.index), len(ds.period)))),
    nse_mm=(["index", "period"], np.zeros((len(ds.index), len(ds.period)))),
    dist_nse_nselog=(["index", "period"], np.zeros((len(ds.index), len(ds.period)))),
    dist_nse_nselog_nsenm7q=(["index", "period"], np.zeros((len(ds.index), len(ds.period)))),
    dist_nse_nselog_nsenm7q_mm=(["index", "period"], np.zeros((len(ds.index), len(ds.period)))),
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

selection_criteria = "dist_nse_nselog_nsenm7q_mm"

print("fill nc with best data")
for period in ds.period:
    for st in ds.index.values:
        print(st)
        if not np.isnan(ds[selection_criteria].sel(index=st, period = period).min().values):        
            pset_best = ds[selection_criteria].sel(index=st, period = period).idxmin().values
            ds_best_st = ds.sel(pset = pset_best, index=st, period=period)
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

#%% plots signature all and best, cumulative, monthly regime, low flows, high flows

start = "2011-01-01"
end = "2021-12-31"

start_hyd = "2011-09-01"
end_hyd = "2021-08-31"

period = "all"

print("make signature plots")
for st in ds.index.values:
    if (st in obs.index) & (not np.isnan(ds[selection_criteria].sel(index=st, period = period).min().values)):
        print(st)
        fig, axes = plt.subplots(2,2, figsize=(12/2.54, 12/2.54))
        ax = axes.flatten()
        #mean monthly flow
        overlap_time = ds_best["Qobs"].sel(index=st, time = slice(start, end)).dropna("time").time.values
        ds["Q"].sel(index=st, time = overlap_time).groupby("time.month").mean().plot(ax=ax[0], hue = "pset", add_legend=False, color = "grey")
        ds_best["Q"].sel(index=st, time = overlap_time, period = period).groupby("time.month").mean().plot(ax=ax[0], add_legend=False, color = "orange")
        ds_best["Qobs"].sel(index=st, time = overlap_time).groupby("time.month").mean().plot(ax=ax[0], add_legend=False, color = "k", linestyle = "--")
        ax[0].set_xticks(np.arange(1,13))
        ax[0].set_xlabel("")
        ax[0].set_ylabel("Q (m$^{3}$ s$^{1}$)", fontsize=fs)

        #cumulative 
        ds["Q"].sel(index=st, time = overlap_time).cumsum("time").plot(ax=ax[1], hue = "pset", add_legend=False, color = "grey")
        ds_best["Q"].sel(index=st, time = overlap_time, period = period).cumsum().plot(ax=ax[1], add_legend=False, color = "orange")
        ds_best["Qobs"].sel(index=st, time = overlap_time).cumsum().plot(ax=ax[1], add_legend=False, color = "k", linestyle = "--")
        ax[1].set_ylabel("Qcum (m$^{3}$ s$^{1}$)", fontsize=fs)

        #lowest flows
        obs_i_nan_nm7q = ds_best["Qobs"].sel(index=st, time = overlap_time).rolling(time = 7).mean().resample(time = 'A').min('time').compute()
        best_i_nan_nm7q = ds_best["Q"].sel(index=st, time = overlap_time, period=period).rolling(time = 7).mean().resample(time = 'A').min('time').compute()

        gumbel_p1, plotting_positions_obs, ymax_obs = get_plotting_position(obs_i_nan_nm7q, ascending=False)
        gumbel_p1, plotting_positions_best, ymax_best = get_plotting_position(best_i_nan_nm7q, ascending=False)
        ymax = np.maximum(ymax_obs, ymax_best) 

        ts = [2., 5.,10.,30.]

        for pset in ds.pset.values:
            sim_i_nan = ds["Q"].sel(index = st, time=overlap_time, pset=pset)
            sim_i_nan_nm7q = sim_i_nan.rolling(time = 7).mean().resample(time = 'A').min('time')
            gumbel_p1, plotting_positions_sim, ymax_sim = get_plotting_position(sim_i_nan_nm7q, ascending=False)
            ax[2].plot(gumbel_p1, plotting_positions_sim, marker = '.', color = 'grey', linestyle = 'None', markersize = 6)

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

        for pset in ds.pset.values:
            sim_i_nan = ds["Q"].sel(index = st, time=overlap_time_max, pset=pset)
            sim_i_nan_max = sim_i_nan.resample(time = 'AS-Sep').max('time')
            gumbel_p1, plotting_positions_sim, ymax_sim = get_plotting_position(sim_i_nan_max)
            ax[3].plot(gumbel_p1, plotting_positions_sim, marker = '.', color = 'grey', linestyle = 'None', markersize = 6)
        
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






#%% plot  hydro
print("plot hydro")
for st in ds.index.values:
    if (st in obs.index) & (not np.isnan(ds[selection_criteria].sel(index=st, period = period).min().values)):
        print(st)
        fig, axes = plt.subplots(2,1, figsize=(12/2.54, 12/2.54),  sharey=True)
        ax = axes.flatten()
        # ds["Q"].sel(index=st, time = slice(start, end)).plot(ax=ax[0], hue = "pset", add_legend=False, color = "grey", linewidth=0.8)
        ds_best["Q"].sel(index=st, time = slice(start, end), period = period).plot(ax=ax[0], add_legend=False, color = "orange", linewidth=0.8)
        ds_best["Qobs"].sel(index=st, time = slice(start, end)).plot(ax=ax[0], add_legend=False, color = "k", linestyle = "--", linewidth=0.8)
        # 2015
        year = 2015
        ds["Q"].sel(index=st, time = slice(f"{year}-01-01", f"{year}-12-31")).plot(ax=ax[1], hue = "pset", add_legend=False, color = "grey", alpha=0.5, linewidth=0.8)
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





#%% dotty plots 
print("plot dotty plots")
period = "all"
for st in ds.index.values:
    if (st in obs.index) & (not np.isnan(ds[selection_criteria].sel(index=st, period = period).min().values)):
        print(st)
        for perf_ind in perf_inds:
            fig, axes = plt.subplots(int(len(params_list)/2),2, figsize=(12/2.54, 12/2.54))
            ax = axes.flatten()
            for i, param in enumerate(params_list):
                ax[i].plot(ds[param], ds[perf_ind].sel(period = "all", index = st), linestyle="None", marker=".", color = "grey")
                ax[i].plot(ds_best[param].sel(index=st, period =period), ds_best[perf_ind].sel(period = "all", index = st), linestyle="None", marker=".", color = "orange")
                ax[i].set_ylim([0,1])
                ax[i].set_xlabel(param, fontsize = fs)
                ax[i].tick_params(axis="both", labelsize=fs)
                ax[i].set_ylabel(f"{perf_ind}", fontsize = fs)
            plt.suptitle(f"{ds_best.sel(index=st).name.values} ({st})", fontsize = fs)
            plt.tight_layout()
            plt.savefig(os.path.join(Folder_plots, f"dottyplots_{ds_best.sel(index=st).name.values}_{st}_{perf_ind}.png"), dpi=300)


