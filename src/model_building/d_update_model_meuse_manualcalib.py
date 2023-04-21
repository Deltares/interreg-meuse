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

#%% base: geul add: manual calib max leakage and soil thickness 

pdir = r"p:\11208719-interreg\wflow"
case_base = "c_geul"
case_update = "d_manualcalib"
run_folder = "run_geul"
config_fn = "run_geul.toml"
run_folder_update = "run_manualcalib"

yml = os.path.join(pdir, "..", "data", "data_meuse.yml")

root = os.path.join(pdir, case_base, run_folder)
mod = WflowModel(root=root, config_fn = config_fn, mode="r", data_libs=yml)

#initialize staticmaps and staticgeoms
mod.staticmaps
mod.staticgeoms


#%%


gdf_org = mod.data_catalog.get_geodataframe(
            "cal_manual"
        )

st_scale = mod.staticmaps.raster.rasterize(gdf_org, "SoilThickness_scale", nodata = -999)
ml_offset = mod.staticmaps.raster.rasterize(gdf_org, "MaxLeakage_offset", nodata = -999)
ml_scale = mod.staticmaps.raster.rasterize(gdf_org, "MaxLeakage_scale", nodata = -999)


soithickness_adapted = st_scale * mod.staticmaps["SoilThickness"]
soithickness_adapted.name = "SoilThickness_manual_cal"

maxleak_adapted = ml_scale * mod.staticmaps["MaxLeakage"] + ml_offset
maxleak_adapted.name = "MaxLeakage_manual_cal"


mod.set_staticmaps(soithickness_adapted)
mod.set_staticmaps(maxleak_adapted)


#%%
mod.staticmaps

mod.set_config("input.vertical.soilthickness", "SoilThickness_manual_cal")
mod.set_config("input.vertical.soilminthickness", "SoilThickness_manual_cal")
mod.set_config("input.vertical.MaxLeakage", "MaxLeakage_manual_cal")
mod.config

#additional changes based on grade calib:
mod.set_config("input.lateral.river.width", 'wflow_riverwidth_sobek_global_extrapol')
mod.set_config("input.lateral.river.bankfull_depth", 'RiverDepth_sobek_global_extrapol')
mod.set_config("input.lateral.subsurface.ksathorfrac", 'ksathorfrac_sub')

mod.set_config("input.lateral.river.floodplain.volume", {'netcdf':{'variable': {'name' :"floodplain_volume"}}, 'scale':2, 'offset':0  }) #"scale":2, "offset": 0})

#new code naming 
# more output 
gauge_toml_header = ["Qriv", "Q"]
gauge_toml_param = ["lateral.river.q_channel_av", "lateral.river.q_av"]
basename = "gauges_Sall"
for o in range(len(gauge_toml_param)):
    gauge_toml_dict = {
                    "header": gauge_toml_header[o],
                    "map": f"gauges_{basename}",
                    "parameter": gauge_toml_param[o],
                }
    mod.config["csv"]["column"].append(gauge_toml_dict)
    
# mod.config({"[input.lateral.river.floodplain]":{"scale":2, "offset": 1}})

mod.set_config("input.lateral.river.floodplain",{"scale":2, "offset": 1})


mod.config["csv"]["column"]

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


