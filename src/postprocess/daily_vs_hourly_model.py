#%% To run with hydromt-wflow
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
# %% Defining functions
def roundup(x):
        return int(math.ceil(x / 100.0)) * 100

def calculate_emp(data):
    emp_p = pd.DataFrame(data=data, columns=['value'])
    emp_p['rank'] = emp_p.iloc[:,0].rank(axis=0, ascending=False, method = 'dense')
    emp_p['exc_prob'] = emp_p['rank']/(emp_p['rank'].size+1) #change this line with what Anaïs sends to me, but is already correct
    emp_p['cum_prob'] = 1 - emp_p['exc_prob']
    emp_p['emp_rp'] = 1/emp_p['exc_prob']
    return emp_p
#%% We import the modelled data
Folder_start = "/p/11208719-interreg"
model_wflow = "f_spwgauges"
Folder_p = os.path.join(Folder_start, "wflow", model_wflow)
folder = "members_bias_corrected_daily"
dt = folder.split("_")[-1]
fn_fig = os.path.join(Folder_start, "Figures", model_wflow)
#%%
#We import the Wflow model
fn_config = os.path.join(Folder_p, folder, 'r10i1p5f1', 'members_bias_corrected_daily_r10i1p5f1.toml')
mod = WflowModel(root=os.path.join(Folder_p), mode="r+", config_fn=fn_config)
#mod.staticgeoms.keys()
#mod.staticmaps

if not os.path.exists(fn_fig):
    os.makedirs(fn_fig)

fn_runs = glob.glob(os.path.join(Folder_p, folder, '*', 'output.csv'))
date_parser = lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S")

model_runs = {
    "r1": {"case":folder, 
            "folder": "r1i1p5f1"},
    "r2": {"case":folder, 
            "folder": "r2i1p5f1"},
    "r3": {"case":folder, 
            "folder": "r3i1p5f1"},
    "r4": {"case":folder, 
            "folder": "r4i1p5f1"}, 
    "r5": {"case":folder, 
            "folder": "r5i1p5f1"}, 
    "r6": {"case":folder, 
            "folder": "r6i1p5f1"}, 
    "r7": {"case":folder, 
            "folder": "r7i1p5f1"}, 
    "r8": {"case":folder, 
            "folder": "r8i1p5f1"}, 
    "r9": {"case":folder, 
            "folder": "r9i1p5f1"}, 
    "r10": {"case":folder, 
            "folder": "r10i1p5f1"}, 
    "r11": {"case":folder, 
            "folder": "r11i1p5f1"}, 
    "r12": {"case":folder, 
            "folder": "r12i1p5f1"}, 
    "r13": {"case":folder, 
            "folder": "r13i1p5f1"}, 
    "r14": {"case":folder, 
            "folder": "r14i1p5f1"}, 
    "r15": {"case":folder, 
            "folder": "r15i1p5f1"}, 
    "r16": {"case":folder, 
            "folder": "r16i1p5f1"}, 
} 

#%%We list the gauges and their areas
gauges_names = list(mod.staticgeoms.keys())
gauges_name = [i for i in gauges_names if i.startswith('gauges_S')] #We just need Sall?
areas_name = [i for i in gauges_names if i.startswith('subcatch_S')]
areas_name.remove('subcatch_Sall')
gauges_name.remove('gauges_Sall')
#%%
df_areas = gpd.GeoDataFrame()
for area in areas_name:
    print(area)
    ds = mod.staticgeoms[area]
    ds_layer = ds.columns[0]
    ds.rename(columns={ds.columns[0]:'Q_'}, inplace = True)
    ds['wflow_subcatch'] = ds_layer
    df_areas = pd.concat([df_areas, ds], ignore_index = True)
#    ids = np.unique(ds.values)
df_areas = gpd.GeoDataFrame(df_areas, crs = ds.crs)
df_areas['area_km2'] = df_areas.to_crs(3044).area/1000000

fn_out = os.path.join(fn_fig, 'output', 'df_areas.geojson')
df_areas.to_file(fn_out, driver="GeoJSON")  
#%%
locs = [f'Q_{int(i)}' for i in df_areas['Q_'].values]
use_cols = ['time']+ [f'Q_{int(i)}' for i in df_areas['Q_'].values]
# %%

#%%We load the daily data 
# Storing all the results in one location 
# runs_dict_daily = {}
# for key in model_runs.keys():
#     print(key)
#     case = model_runs[key]["case"]
#     ens = model_runs[key]["folder"] 
#     runs_dict_daily[key] = pd.read_csv(os.path.join(Folder_p, case, ens, "output.csv"), index_col=['time'], header=0, usecols = use_cols, parse_dates=['time'], date_parser = date_parser)

# #We load the hourly data 
# runs_dict_hourly = {}
# for key in model_runs.keys():
#     print(key)
#     case = "members_bias_corrected_hourly"
#     ens = model_runs[key]["folder"] 
#     runs_dict_hourly[key] = pd.read_csv(os.path.join(Folder_p, case, ens, "output.csv"), index_col=['time'], header=0, usecols = use_cols, parse_dates=['time'], date_parser = date_parser)
# #%%
# runs_dict_all= {}
# runs_dict_all['daily'] = runs_dict_daily
# runs_dict_all['hourly'] = runs_dict_hourly

# %%
#for dt in ['daily', 'hourly']:
dt = 'hourly'

runs_dict = {}
for key in model_runs.keys():
    print(key)
    case = f"members_bias_corrected_{dt}"
    ens = model_runs[key]["folder"] 
    runs_dict[key] = pd.read_csv(os.path.join(Folder_p, case, ens, "output.csv"), index_col=['time'], header=0, usecols = use_cols, parse_dates=['time'], date_parser = date_parser)
#%%
ds_all = xr.Dataset()
for station in locs:
        print(station)
        print(dt)
        #For now we select one and extract the extremes
        ts_peaks=pd.DataFrame()
        ts_dates = pd.DataFrame()
        i = 0
        for key in model_runs.keys():
                #print(i, key)
                ts_key = runs_dict[key][station]
                #We perform the EVA - BM - Gumbel 
                peaks = pyex.get_extremes(ts_key, method = "BM", extremes_type="high", block_size="365.2426D", errors="raise")
                #Storing the date from the original AM time series 
                dates = pd.DataFrame(columns=['month','date', 'member'])
                dates['month'] = peaks.index.month
                dates['date'] = peaks.index
                dates['member'] = key

                peaks.index  = peaks.index + pd.DateOffset(years=int(i)) 
                ts_peaks = pd.concat([ts_peaks, peaks], axis = 0)
                ts_dates = pd.concat([ts_dates, dates], axis = 0, ignore_index = True)
                
                #print(len(ts_peaks))
                i += len(ts_key.index.year.unique())
        ts_peaks.rename(columns={0:station}, inplace = True)
        if len(ts_peaks) == 1040:
                print("Date conversion seems ok!")
        #ts_peaks.sort_index(axis = 0, ascending=True, inplace = True)

        emp_T = calculate_emp(ts_peaks.values)
        df_all = pd.concat([emp_T, ts_dates], axis = 1)
        df_all.reset_index(drop=False, inplace = True)
        df_all = df_all.rename(columns={df_all.columns[0]:'year_i'})
        df_all.sort_values(by='value', ascending = False, inplace=True)
        df_all['area_km2'] = df_areas.set_index('Q_').loc[float(station.strip("Q_")),'area_km2']

        #We store it as a Dataset
        ds_try = df_all.to_xarray()
        ds_try = ds_try.expand_dims(dim={"Q":[station]})
        ds_all = xr.merge([ds_all,ds_try])   
fn_out = os.path.join(fn_fig, 'output', f'AM_{dt}.nc')
ds_all.to_netcdf(fn_out)
#%%
#Ts = [1.1, 1.5, 1.8, 2, 5, 10, 25, 50, 100, 250, 500, 1000]

