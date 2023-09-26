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

def intersection(a, b):
    return list(set(a).intersection(b))

#%% read model and observations 
fs = 8

#windows path
Folder_plots = r"p:\11208719-interreg\wflow\j_waterschaplimburg\runs_calibration_linux_01\Results\Plots"
Folder_plots_fr = r"p:\11208719-interreg\wflow\j_waterschaplimburg\runs_calibration_linux_01\Results\Plots_fr"
Folder_plots_d = r"d:\interreg\Plots\calibration"
root = r"p:\11208719-interreg\wflow\j_waterschaplimburg\runs_calibration_linux_01"
config_folder = r"n:\My Documents\unix-h6\interreg-meuse\src\model_building\calibration_linux\config"
obs_catalog = os.path.join(config_folder,"spw_windows.yml")


print("Reading wflow model")
toml_default_fn = "wflow_sbm_calibration.toml"
mod = WflowModel(root, config_fn=toml_default_fn, data_libs = obs_catalog, mode="r")

# ds = xr.open_dataset(os.path.join(Folder_plots, "ds_output.nc"))

ds_best = xr.open_dataset(os.path.join(Folder_plots, "ds_output_best.nc"))
ds_best_fr = xr.open_dataset(os.path.join(Folder_plots_fr, "ds_output_best.nc"))
ds_best_all = xr.merge([ds_best, ds_best_fr])


perf_inds = ["nse", "nse_log", "kge", "nse_nm7q", "nse_maxq", "nse_mm", "nse_cum", "dist_nse_nselog", "dist_nse_nselog_nsenm7q", "dist_nse_nselog_nsenm7q_mm", "dist_nse_nselog_nsenm7q_mm_maxq", "dist_nse_nselog_nsenm7q_mm_cum", "dist_nse_nselog_nsenm7q_mm_maxq_cum"]

params = ['ksathorfrac',
 'maxleakage',
 'soilthickness',
 'rootingdepth',
 'storagewood',
 'n',
 'soilminthickness']

dic_params = {"ksathorfrac": {"netcdf_name" : ["ksathorfrac_sub"], 
                              "type" : "scale", #scale, offset or value
                              "default": 1,
                                } ,
              "maxleakage": {"netcdf_name" : ["MaxLeakage"], 
                              "type" : "value", #scale, offset or value
                              "default": 0,
                                } ,
              "soilthickness": {"netcdf_name" : ["SoilThickness"], 
                              "type" : "scale", #scale, offset or value
                              "default": 1,
                                } ,
            #   "soilminthickness": "SoilThickness",
              "rootingdepth": {"netcdf_name" : ["RootingDepth_obs_20"], 
                              "type" : "scale", #scale, offset or value
                              "default": 1,
                                } ,
            #NB: n changes 2 maps!
              "n": {"netcdf_name" : ["N", "N_River"],
                              "type" : "scale", #scale, offset or value
                              "default": 1,
                                } ,
              "storagewood":  {"netcdf_name" : ["Swood"], 
                              "type" : "offset", #scale, offset or value
                              "default": 0,
                                } ,

              }

#spw stations
obs_fn = "spw"
obs_ts_fn_daily_sel = "meuse-hydro_timeseries_spw_selection100km2_Q" #this is daily!
obs_sel = mod.data_catalog.get_geodataset(obs_ts_fn_daily_sel, geom=mod.basins)
index_name = "id"

#french stations
obs_fn_fr = "Sall"
obs_ts_fn_fr = "meuse-hydro_timeseries_hydroportail_hourly"
obs_fr = mod.data_catalog.get_geodataset(obs_ts_fn_fr, geom=mod.basins)
index_name = "wflow_id"

#%% print best param values

for param in params:
    print(ds_best[param].sel(period = "all").dropna("index").values)

#%% print perf indicator values
for perf_ind in perf_inds:
    print(ds_best[perf_ind].sel(period = "all").dropna("index").values)

#%% add to staticgeoms gauges (100km  before selection)

rivers = mod.staticgeoms["rivers"]

gdf_gauges_all = mod.staticgeoms[f"gauges_{obs_fn}"]
gdf_gauges_all = gdf_gauges_all.set_index("id")

stations = gdf_gauges_all.index.values
#select subset
stations_sel = intersection(obs_sel.index.values, gdf_gauges_all.index.values)
gdf_gauges_all = gdf_gauges_all.loc[stations_sel]
#check which subset was really used in calibration
idx = ds_best["nse"].sel(period="all").dropna(dim="index").index
gdf_gauges_cal = gdf_gauges_all.loc[idx]
#drop chooz as it is done in the french part...
gdf_gauges_cal = gdf_gauges_cal.drop(8702)

#now french stations:
gdf_gauges_fr = mod.staticgeoms[f"gauges_{obs_fn_fr}"]
gdf_gauges_fr = gdf_gauges_fr.set_index("wflow_id")
gdf_gauges_fr.index.name = "id"
#only small subset in calibration 
idx = ds_best_fr["nse"].sel(period="all").dropna(dim="index").index
gdf_gauges_cal_fr = gdf_gauges_fr.loc[idx]

#%% #make subcatch map with all catch from calibration - combine french and belgium stations! 

gdf_gauges_spw_fr = pd.concat([gdf_gauges_cal, gdf_gauges_cal_fr], axis=0)


#%% #make subcatch map with all catch from calibration - spw and fr


#test geodataframe 
gauges_calib_geoj_fn = os.path.join(Folder_plots_d, "stations_cal.geojson")
gdf_gauges_spw_fr.to_file(gauges_calib_geoj_fn)

# gdf_gauges_cal = pd.concat([gdf_gauges_cal.geometry.x.rename("lon"), gdf_gauges_cal.geometry.y.rename("lat"), gdf_gauges_cal[["nom", "bassin versant", "superficie bv km2", ]],],axis=1)
# gdf_gauges_cal.index.name = "id"
# gauges_calib_csv_fn = os.path.join(Folder_plots_d, "stations_cal.csv")
# gdf_gauges_cal.to_csv(gauges_calib_csv_fn)


basename = "stations_obs_cal"
#same as below but then with geojson. 
mod.setup_gauges(gauges_fn=gauges_calib_geoj_fn,
                 index_col="id",
                 derive_subcatch=True,
                 basename=basename,
                 )

#test gauge branch ! 
# # basename = "stations_obs_cal"
# # mod.setup_gauges(gauges_fn=gauges_calib_csv_fn,
# #                  index_col="id",
# #                  derive_subcatch=True,
# #                  basename=basename,
# #                  )

#you don't need to write to file! can also directly read in a geodataframe 
# mod.setup_gauges(gauges_fn=gdf_gauges_cal,
#                  index_col="id",
#                  derive_subcatch=True,
#                  basename=basename,
#                  )
# #test area snap 
# basename = "snap_area"
# mod.setup_gauges(gauges_fn="d:\hydromt\gauges\Stations_Q_jour.csv",
#                  index_col="id",
#                  derive_subcatch=True,
#                  basename=basename,
#                  snap_uparea = True,
#                  )

# basename = "snap_river"
# mod.setup_gauges(gauges_fn="d:\hydromt\gauges\Stations_Q_jour.csv",
#                  index_col="id",
#                  derive_subcatch=True,
#                  basename=basename,
#                  snap_uparea = False,
#                  )

# fig, ax = plt.subplots()
# mod.staticgeoms["subcatch_snap_river"].plot(edgecolor = "k",ax=ax)

# fig, ax = plt.subplots()
# mod.staticgeoms["subcatch_snap_area"].plot(edgecolor = "k",ax=ax)

# root_updated = r"d:\hydromt\gauges\test_model"
# mod.set_root(root_updated)
# mod.write_staticmaps()
# mod.write_staticgeoms()

# wdw = 3
# ds_wdw = mod.staticmaps.raster.sample(gdf_gauges_cal, wdw=wdw)
# mod.staticmaps["wflow_gauges_snap_river"]



# #test branch
# mod.setup_outlets(river_only=False)
# fig, ax = plt.subplots()
# mod.staticgeoms["basins"].plot(edgecolor = "k",ax=ax)
# mod.staticgeoms["gauges"].plot(edgecolor = "k",ax=ax)
# mod.setup_config_output_timeseries(mapname = "wflow_gauges_stations_obs_cal", toml_output = "netcdf", header = ["Q_haha", "Q_hihi"], param = ["lateral.river.q_av", "lateral.river.q_av"])
# mod.setup_config_output_timeseries(mapname = "wflow_gauges_stations_obs_cal", toml_output = "netcdf", header = ["Q_haha", "Q_hihi"], param = ["lateral.river.q_av", "lateral.river.q_av"])
# mod.config



#%% add to subcatch and gauges cal map 

gdf_gauges = mod.staticgeoms["gauges_stations_obs_cal"]
# gdf_gauges_all = gdf_gauges_all.set_index("id")


gdf_sub = mod.staticgeoms["subcatch_stations_obs_cal"]
gdf_sub = gdf_sub.set_index("value")
gdf_sub.index = gdf_sub.index.astype(int)

#merge with ds_best (using ds_best_all to have both spw and fr !! ) make sure Chooz is not included from SPW side
for perf_ind in perf_inds:
    gdf_gauges = pd.concat([gdf_gauges, ds_best_all[perf_ind].sel(period = "all", index= gdf_gauges.index).dropna("index").to_dataframe()], axis=1)
    gdf_sub = pd.concat([gdf_sub, ds_best_all[perf_ind].sel(period = "all", index= gdf_gauges.index).dropna("index").to_dataframe()], axis=1)


for param in params:
    gdf_gauges = pd.concat([gdf_gauges, ds_best_all[param].sel(period = "all", index= gdf_gauges.index).dropna("index").to_dataframe()], axis=1)
    gdf_sub = pd.concat([gdf_sub, ds_best_all[param].sel(period = "all", index= gdf_gauges.index).dropna("index").to_dataframe()], axis=1)

#%%
# gdf_gauges.plot(column = "dist_nse_nselog_nsenm7q_mm", vmin = 0, vmax = 1, legend=True)

for perf_ind in perf_inds:
    fig, ax = plt.subplots()
    rivers.plot(ax=ax, linewidth = 0.8)
    gdf_gauges.plot(column = perf_ind, vmin = 0, vmax = 1, legend=True, ax=ax)
    plt.title(f"{perf_ind}")

for param in params:
    fig, ax = plt.subplots()
    rivers.plot(ax=ax, linewidth = 0.8)
    gdf_gauges.plot(column = param, legend=True, ax=ax)
    plt.title(f"{param}")

for perf_ind in perf_inds:
    fig, ax = plt.subplots()
    rivers.plot(ax=ax, linewidth = 0.8)
    gdf_sub.plot(column = perf_ind, vmin = 0, vmax = 1, legend=True, ax=ax)
    plt.title(f"{perf_ind}")

for param in params:
    fig, ax = plt.subplots()
    rivers.plot(ax=ax, linewidth = 0.8)
    gdf_sub.plot(column = param, legend=True, ax=ax)
    plt.title(f"{param}")


#%% regionalization for empty parts and add to staticmaps

#regionalization through nearest neighbor interpolation of multiplication factor map. 
#make new parameter maps and add to staticmaps

#no interpolation 
interp = False

for param in dic_params:
    nc_names = dic_params[param]["netcdf_name"]
    type = dic_params[param]["type"]
    default = dic_params[param]["default"]
    for nc_name in nc_names:
        print(param, nc_name, type, default)
        if interp == True:
            da = mod.staticmaps.raster.rasterize(gdf_sub, col_name = param, nodata=-999)
            da_interp = da.raster.interpolate_na(method="nearest").where(mod.staticmaps["wflow_subcatch"]>0, -999)
            da_interp.raster.set_nodata(-999)
        else:
            da_interp = mod.staticmaps.raster.rasterize(gdf_sub, col_name = param, nodata=default)

        if type == "scale": # (param == "ksathorfrac") | (param=="rootingdepth") | (param=="soilthickness"):
            da_calibrated = da_interp * mod.staticmaps[nc_name]
            da_calibrated = da_calibrated.where(mod.staticmaps["wflow_subcatch"]>0, -999)
            da_calibrated.raster.set_nodata(-999)
        # elif param == "n":
        #     for nc_name in ["N", "N_River"]:
        #         da_calibrated = da_interp * mod.staticmaps[nc_name]
        #         da_calibrated = da_calibrated.where(mod.staticmaps["wflow_subcatch"]>0, -999)
        #         da_calibrated.raster.set_nodata(-999)
        #         da_calibrated.name = f"{nc_name}_cal"
        elif type == "offset": # param == "storagewood": #offset
            da_calibrated = da_interp + mod.staticmaps[nc_name]
            da_calibrated = da_calibrated.where(mod.staticmaps["wflow_subcatch"]>0, -999)
            da_calibrated.raster.set_nodata(-999)
        else: #value
            da_calibrated = da_interp
            da_calibrated = da_calibrated.where(mod.staticmaps["wflow_subcatch"]>0, -999)
            da_calibrated.raster.set_nodata(-999)
        da_calibrated.name = f"{nc_name}_cal"
        print(f"{nc_name}_cal")
        mod.set_staticmaps(da_calibrated)

#check
for param in dic_params:
    nc_names = dic_params[param]["netcdf_name"]
    for nc_name in nc_names:
        print(param, nc_name)
        if param == "floodplainvolume":
            plt.figure(); mod.staticmaps[f"{nc_name}_cal"].rename(f"{param}").sel(flood_depth=0.5).raster.mask_nodata().plot()
        else:
            plt.figure(); mod.staticmaps[f"{nc_name}_cal"].rename(f"{param}").raster.mask_nodata().plot()
        plt.title("")
        plt.tight_layout()
        # plt.savefig(os.path.join(Folder_plots_d, f"{param}_{nc_name}_cal.png"), dpi=300)
        # plt.savefig(os.path.join(Folder_plots_d, f"{param}_{nc_name}_cal_02.png"), dpi=300)
        plt.savefig(os.path.join(Folder_plots_d, f"{param}_{nc_name}_cal_03.png"), dpi=300)


#%% just set path of staticmaps right in toml 

# mod.set_config("input.path_static", "staticmaps.nc")
# mod.set_config("input.path_forcing", "../wflow_wallonie/eobs_v25.0e_1980_2020.nc")
# mod.config

#%% write calibrated model 
# root_updated = r"p:\11208719-interreg\wflow\k_snakecal\run_snakecal"
# root_updated = r"p:\11208719-interreg\wflow\l_snakecal02\run_snakecal02"
root_updated = r"p:\11208719-interreg\wflow\m_snakecal03\run_snakecal03"
mod.set_root(root_updated)
mod.write_staticmaps()
mod.write_staticgeoms()

#update toml  manually from original model, not from calibration toml!



#%%run model with different linux settings. 


