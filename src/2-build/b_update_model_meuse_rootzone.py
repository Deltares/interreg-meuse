# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 13:44:29 2023

@author: bouaziz
"""

import hydromt
import xarray as xr
import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
from hydromt_wflow import WflowModel

#%% base: floodplain1d add: rootzoneDepth

pdir = r"p:\11208719-interreg\wflow"
case_base = "a_floodplain1d"
case_update = "b_rootzone"
run_folder = "run_default"
config_fn = "wflow_sbm.toml"
run_folder_update = "run_rootingdepth_rp_20"

yml = os.path.join(pdir, "..", "data", "data_meuse.yml")

root = os.path.join(pdir, case_base, run_folder)
mod = WflowModel(root=root, config_fn = config_fn, mode="r", data_libs=yml)

#initialize staticmaps and staticgeoms
mod.staticmaps
mod.staticgeoms

#%%for now in branch rootzone_clim
mod.setup_rootzoneclim(run_fn="meuse-hydro_timeseries", 
                        forcing_obs_fn="inmaps_meuse_genre", #already converted to daily 
                        start_hydro_year = "Oct",
                        start_field_capacity = "Apr",                       
                        time_tuple = ("2005-01-01", "2017-12-31"), 
                        missing_days_threshold=330,
                        return_period = [2,5,10,15,20]
                        )
#%%
mod.staticmaps

mod.set_config("input.vertical.rootingdepth", "RootingDepth_obs_20")
mod.config

root_updated = os.path.join(pdir, case_update, run_folder_update)
mod.set_root(root_updated)

mod.write_staticmaps()
mod.write_staticgeoms()
mod.write_config(config_name=f"{run_folder_update}.toml")

#remove run_default dir
if len(os.listdir(os.path.join(pdir, case_update, run_folder_update , 'run_default'))) == 0:
        os.rmdir(os.path.join(pdir, case_update, run_folder_update , 'run_default'))

#remove instate dir
if len(os.listdir(os.path.join(pdir, case_update, run_folder_update , 'instate'))) == 0:
        os.rmdir(os.path.join(pdir, case_update, run_folder_update , 'instate'))

#move staticgeoms
shutil.move(os.path.join(pdir, case_update, run_folder_update , 'staticgeoms'),
            os.path.join(pdir, case_update, 'staticgeoms'),
            )

#write run bat file
path_env_julia = r"c:\Users\bouaziz\.julia\environments\wflow_floodplain1d"

bat_str = f"""
julia  -t 4 --project={path_env_julia} -e "using Wflow; Wflow.run()" "{run_folder_update}\{run_folder_update}.toml"
"""

bat_file = os.path.join(pdir, case_update, f"{run_folder_update}.cmd")
with open(bat_file, mode="w") as f:
    f.write(bat_str)

# plt.figure(); mod.staticmaps["rootzone_storage_obs_20"].plot()
# plt.figure(); mod.staticmaps["RootingDepth_obs_20"].plot(vmin=0, vmax=1400)
# plt.figure(); mod.staticmaps["RootingDepth"].raster.mask_nodata().plot(vmin=0, vmax=1400)
# plt.figure(); (mod.staticmaps["RootingDepth_obs_20"]/mod.staticmaps["RootingDepth"]).plot()
