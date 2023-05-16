import hydromt
import xarray as xr
import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
from hydromt_wflow import WflowModel
import glob

pdir = r"p:\11208719-interreg\wflow"
case_base = "h_spwgauges"
case_update = "i_hydroportail"
run_folder = "run_spwgauges"
config_fn = "run_spwgauges.toml"
run_folder_update = "run_hydroportail"

root = os.path.join(pdir, case_base, run_folder)
mod = WflowModel(root=root, config_fn=config_fn, mode="r")

#initialize staticmaps and staticgeoms
mod.staticmaps
mod.staticgeoms

gauges_path_hydroportail = r"p:\11208719-interreg\data\hydroportail\c_final\hydroportail_stations.csv"
basename = "hydroportail"
gauge_toml_header = ["Qriv_hp", "Q_hp"]
gauge_toml_param = ["lateral.river.q_channel_av", "lateral.river.q_av"]

for o in range(len(gauge_toml_param)):
    gauge_toml_dict = {
                    "header": gauge_toml_header[o],
                    "map": f"gauges_{basename}",
                    "parameter": gauge_toml_param[o],
                }
    mod.config["csv"]["column"].append(gauge_toml_dict)

mod.setup_gauges(gauges_fn=gauges_path_hydroportail, 
                 index_col="wflow_id", 
                 derive_subcatch=True, 
                 basename=basename,
                 gauge_toml_header=gauge_toml_header, 
                 gauge_toml_param=gauge_toml_param)

root_updated = os.path.join(pdir, case_update, run_folder_update)
mod.set_root(root_updated)

mod.write_staticmaps()
mod.write_staticgeoms()
mod.write_config(config_name=f"{run_folder_update}.toml")

#copy of lake files 
lake_files = glob.glob(os.path.join(pdir, case_base, "lake*.csv"))
if len(lake_files)>0:
    for lake_file in lake_files:
        dst_lake_file = os.path.join(pdir, case_update, os.path.basename(lake_file))
        shutil.copy(lake_file, dst_lake_file)

# remove run_default dir
if len(os.listdir(os.path.join(pdir, case_update, run_folder_update, 'run_default'))) == 0:
    os.rmdir(os.path.join(pdir, case_update, run_folder_update, 'run_default'))

# remove instate dir
if len(os.listdir(os.path.join(pdir, case_update, run_folder_update, 'instate'))) == 0:
    os.rmdir(os.path.join(pdir, case_update, run_folder_update, 'instate'))

# move staticgeoms
shutil.move(os.path.join(pdir, case_update, run_folder_update, 'staticgeoms'),
            os.path.join(pdir, case_update, 'staticgeoms'),
            )

# write run bat file
path_env_julia = r"c:\Users\bouaziz\.julia\environments\wflow_floodplain1d"
# path_env_julia = r"c:\Users\riveros\.julia\environments\v1.6"

bat_str = f"""
julia  -t 4 --project={path_env_julia} -e "using Wflow; Wflow.run()" "{run_folder_update}\{run_folder_update}.toml"
"""

bat_file = os.path.join(pdir, case_update, f"{run_folder_update}.cmd")
with open(bat_file, mode="w") as f:
    f.write(bat_str)