#%%
import pandas as pd
import matplotlib.pyplot as plt
import pyextremes as pyex
from datetime import datetime
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, glob
from scipy.stats import gumbel_r, genextreme
import xarray as xr
import geopandas as gpd
from hydromt_wflow import WflowModel
import matplotlib.pyplot as plt
from datetime import date, timedelta   

#%% We load one example of the results
# We import the modelled data
Folder_start = "/p/11208719-interreg"
model_wflow = "j_waterschaplimburg"
Folder_p = os.path.join(Folder_start, "wflow", model_wflow)
folder = "members_bias_corrected_daily"
fn_fig = os.path.join(Folder_start, "Figures", model_wflow, folder)

date_parser = lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S")

#%%
fn = r'/p/11208719-interreg/wflow/j_waterschaplimburg/staticmaps.nc'
ds = xr.open_dataset(fn)
#%% For now we pick one and see what's there
ens_i = = 'r1i1p5f1'

fn_csv = 
df =  