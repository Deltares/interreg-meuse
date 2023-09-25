#%%
import pandas as pd
import matplotlib.pyplot as plt
import pyextremes as pyex
from datetime import datetime
import numpy as np
from matplotlib.pyplot import cm
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, glob
from scipy.stats import gumbel_r, genextreme
import xarray as xr
import math
from matplotlib.pyplot import cm
import hydromt
from hydromt_wflow import WflowModel
import geopandas as gpd
import pickle
import sys
from eva_analysis_functions import *


#%% We import the modelled data
Folder_start = "/p/11208719-interreg"
model_wflow = "p_geulrur"#"o_rwsinfo"
Folder_p = os.path.join(Folder_start, "wflow", model_wflow)
folder = "members_bias_corrected_revised_daily" #"members_bias_corrected_revised_hourly"#"members_bias_corrected_revised_daily" #"members_bias_corrected_hourly"
#folder = sys.argv[1] 
fn_fig = os.path.join(Folder_start, "Figures", model_wflow, folder)
fn_data_out = os.path.join(fn_fig, 'data')

dt = folder.split("_")[-1]
print(f"Performing analysis for {dt}")

if not os.path.exists(fn_fig):
    os.makedirs(fn_fig)

#We import the dictionary with the results
fn_out = os.path.join(fn_data_out, f'{dt}_results_stations.pickle')
with open(fn_out, 'rb') as f:
    all_results = pickle.load(f)
#%%
dict_cases = {
        "report_main":  
                {'Rur at Monschau': {'lat': 50.55000 , 'lon': 6.25250, 'id': 15300002, 'source': 'hygon'},
                 'Geul at Meersen': {'lat': 50.89167, 'lon': 5.72750, 'id': 1036, 'source': 'wl'},
                 'Meuse at Goncourt': {'lat': 48.24167, 'lon': 5.61083, 'id': 1022001001, 'source': 'hp'},
                 'Vesdre at Chaudfontaine': {'lat': 50.59167, 'lon': 5.65250, 'id': 6228, 'source': 'spw'},
                 'Ourthe at Tabreux': {'lat': 50.44167, 'lon': 5.53583, 'id': 5921, 'source': 'spw'},
                 'Sambre at Salzinne': {'lat': 50.45833, 'lon': 4.83583, 'id': 7319, 'source': 'spw'},
                 'Meuse at Chooz': {'lat': 50.09167, 'lon': 4.78583, 'id':1720000001, 'source': 'hp'},
                 'Meuse at St Pieter': {'lat': 50.85000 , 'lon': 5.69417, 'id': 16, 'source': 'rwsinfo'},},
        "report_appendix": 
                {'Geul at Hommerich': {'lat': 50.80000, 'lon': 5.91917, 'id': 1030, 'source': 'wl'},
                 'Viroin Treignes': {'lat': 50.09167, 'lon': 4.67750, 'id': 9021, 'source': 'spw'},
                 'Ambleve at Martinrive': {'lat': 50.48333, 'lon': 5.63583, 'id': 6621, 'source': 'spw'},
                 'Semois at Membre Pont': {'lat': 49.86667, 'lon': 4.90250, 'id': 9434, 'source': 'spw'},
                 'Lesse at Gendron': {'lat': 50.20833, 'lon': 4.96083, 'id': 8221, 'source': 'spw'},
                 'Rur at Stah': {'lat': 51.1, 'lon': 6.10250, 'id': 91000001, 'source': 'hygon'},
                 'Meuse at St-Mihiel': {'lat': 48.86667, 'lon': 5.52750, 'id': 12220010, 'source': 'france'},},
}
#%%
Ts = [1.1,1.5,1.8,2.0,5.0,10.0,25.0,50.0,100.0,250.0,500.0,1000.0,1200.0]
all_stations = list()
for case in dict_cases.keys():
        for station_name in dict_cases[case].keys():
                print(station_name)
                all_stations.append(station_name)
#%%
ret_levels = pd.DataFrame(index=all_stations, columns=[Ts])
ci_low = pd.DataFrame(index=all_stations, columns=[Ts])
ci_high = pd.DataFrame(index=all_stations, columns=[Ts])
for case in dict_cases.keys():
        for station_name in dict_cases[case].keys():
                print(station_name)
                source = dict_cases[case][station_name]['source']
                station = dict_cases[case][station_name]['id']

                ret_levels.loc[station_name,:] = all_results[dt][source]['results'][station]["MOD"]['BM_fit']['1024']['return_levels']['genextreme'].values
                ci_low.loc[station_name,:] = all_results[dt][source]['results'][station]["MOD"]['BM_fit']['1024']['conf_int_low']['genextreme'].values
                ci_high.loc[station_name,:] = all_results[dt][source]['results'][station]["MOD"]['BM_fit']['1024']['conf_int_high']['genextreme'].values

ret_levels.to_csv(os.path.join(fn_data_out, f'{dt}_GEV_return_levels_AMs.csv'))
ci_low.to_csv(os.path.join(fn_data_out, f'{dt}_GEV_CI_low_return_levels_AMs.csv'))
ci_high.to_csv(os.path.join(fn_data_out, f'{dt}_GEV_CI_high_return_levels_AMs.csv'))
# %%
