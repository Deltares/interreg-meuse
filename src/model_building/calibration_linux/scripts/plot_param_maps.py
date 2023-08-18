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
Folder_plots_d = r"d:\interreg\Plots\calibration__params_maps_p_geulrur"

print("Reading wflow model")
root = r"p:\11208719-interreg\wflow\p_geulrur\run_geulrur"
toml_fn = "run_geulrur.toml"
mod = WflowModel(root, config_fn=toml_fn, mode="r")

mod.staticmaps

#%%

dic_params = {"ksathorfrac [-]": {"netcdf_name" : ["ksathorfrac_sub_cal"], 
                              "type" : "scale", #scale, offset or value
                              "default": 1,
                                } ,
              "maxleakage [mm d$^{-1}$]": {"netcdf_name" : ["MaxLeakage_cal"], 
                              "type" : "value", #scale, offset or value
                              "default": 0,
                                } ,
              "soilthickness [mm]": {"netcdf_name" : ["SoilThickness_cal"], 
                              "type" : "scale", #scale, offset or value
                              "default": 1,
                                } ,
            #   "soilminthickness": "SoilThickness",
              "rootingdepth [mm]": {"netcdf_name" : ["RootingDepth_obs_20_cal"], 
                              "type" : "scale", #scale, offset or value
                              "default": 1,
                                } ,
            #NB: n changes 2 maps!
              "n [s m$^{-1/3}$]": {"netcdf_name" : ["N_cal", "N_River_cal"],
                              "type" : "scale", #scale, offset or value
                              "default": 1,
                                } ,
              "storagewood [mm]":  {"netcdf_name" : ["Swood_cal"], 
                              "type" : "offset", #scale, offset or value
                              "default": 0,
                                } ,

              }

#%%
#plot
for param in dic_params:
    nc_names = dic_params[param]["netcdf_name"]
    for nc_name in nc_names:
        print(param, nc_name)
        fig, ax  = plt.subplots()
        mod.staticmaps[f"{nc_name}"].rename(f"{param}").raster.mask_nodata().plot(ax=ax)
        #add geoms
        mod.staticgeoms["subcatch_S06"].plot(ax=ax, edgecolor="r", facecolor="None")
        mod.staticgeoms["subcatch_hygon_91000001"].plot(ax=ax, edgecolor="k", facecolor="None")
        mod.staticgeoms["subcatch_waterschaplimburg_1036"].plot(ax=ax, edgecolor="k", facecolor="None")
        plt.title("")
        plt.tight_layout()
        # plt.savefig(os.path.join(Folder_plots_d, f"{param}_{nc_name}_cal.png"), dpi=300)
        # plt.savefig(os.path.join(Folder_plots_d, f"{param}_{nc_name}_cal_02.png"), dpi=300)
        param_short_name = param.split(" ")[0]
        plt.savefig(os.path.join(Folder_plots_d, f"{param_short_name}_{nc_name}_p_geulrur.png"), dpi=300)



