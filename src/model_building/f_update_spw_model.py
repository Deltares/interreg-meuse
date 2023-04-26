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
case_base = "d_manualcalib"
case_update = "f_spwgauges"
run_folder = "run_manualcalib"
config_fn = "run_manualcalib.toml"
run_folder_update = "run_spwgauges"

root = os.path.join(pdir, case_base, run_folder)
mod = WflowModel(root=root, config_fn=config_fn, mode="r")

#initialize staticmaps and staticgeoms
mod.staticmaps
mod.staticgeoms

gauges_path_ge20_manual = r"p:\11208186-spw\Data\Debits\Q_jour\Stations_Q_jour_ge20_manual.csv"

# basename = "stations_obs"
basename = "spw"
mod.setup_gauges(gauges_fn=gauges_path_ge20_manual, index_col="id", derive_subcatch=True, basename=basename,)

root_updated = os.path.join(pdir, case_update, run_folder_update)
mod.set_root(root_updated)

mod.write_staticmaps()
mod.write_staticgeoms()
mod.write_config(config_name=f"{run_folder_update}.toml")

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