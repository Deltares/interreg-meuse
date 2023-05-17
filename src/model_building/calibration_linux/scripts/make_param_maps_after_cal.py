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
Folder_plots = r"p:\11208186-spw\Models\Wflow\wflow_wallonie_rivers_gauges_global_fldplain_rz100km2_aardewerk1_lu_riv10\runs_calibration_linux_01\Results\Plots"
root = r"p:\11208186-spw\Models\Wflow\wflow_wallonie_rivers_gauges_global_fldplain_rz100km2_aardewerk1_lu_riv10\runs_calibration_linux_01"
config_folder = r"p:\11208186-spw\src\calibration_linux\config"
obs_catalog = os.path.join(config_folder,"spw_windows.yml")

print("Reading wflow model")
toml_default_fn = "wflow_sbm_calibration.toml"
mod = WflowModel(root, config_fn=toml_default_fn, data_libs = obs_catalog, mode="r")

ds = xr.open_dataset(os.path.join(Folder_plots, "ds_output.nc"))

ds_best = xr.open_dataset(os.path.join(Folder_plots, "ds_output_best.nc"))

