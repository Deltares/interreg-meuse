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
#folder = "members_bias_corrected_revised_hourly" #'members_bias_corrected_revised_daily' #"members_bias_corrected_revised_hourly"#"members_bias_corrected_revised_daily" #"members_bias_corrected_hourly"
folder = sys.argv[1] 
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


#%% read one model 
root = r"/p/11208719-interreg/wflow/p_geulrur"#r"/p/11208719-interreg/wflow/o_rwsinfo"
config_fn = f"{folder}/r11i1p5f1/{folder}_r11i1p5f1.toml"
yml = r"/p/11208719-interreg/data/data_meuse_linux.yml"
mod = WflowModel(root = root, config_fn=config_fn, data_libs=["deltares_data", yml], mode = "r")

#%% We define the dictionary we want to show
dict_cases = {
        "report_main":  
                {"sel_stations_case":[7319, 5921, 6228, 1036, 1022001001, 1720000001, 16, 91000001],},
        "report_appendix": 
                {"sel_stations_case":[6621, 9434, 9021, 8221, 1030, 12220010, 15300002]}
}
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
source_dic_daily = {

    "spw":{"coord":"spw_gauges_spw",
            "fn":"spw_qobs_daily", #entry in datacatalog 
            "fn_stats": None,
            "stations": [7319, 5921, 6228, 6621, 9434, 9021, 8221],},

    "wl":{"coord":"wl_gauges_waterschaplimburg",
            "fn":"wl_qobs_daily", 
            "fn_stats": None,
            "stations": [1036, 1030],},

    "hp":{"coord":"hp_gauges_hydroportail",
            "fn": "hp_qobs_daily", 
            "fn_stats": "hp_qstats_daily",            
            "stations": [1022001001, 1720000001],},

    "rwsinfo":{"coord":"rwsinfo_gauges_rwsinfo",
            "fn":"rwsinfo_qobs_daily", 
            "fn_stats": "rwsinfo_qstats_daily",   
            "stations": [16],},    

    "france":{"coord":"france_gauges_france",
            "fn":"france_qobs_daily", 
            "fn_stats": "hp_qstats_daily",   
            "stations": [12220010],},    

    "hygon":{"coord":"hygon_gauges_hygon",
            "fn":"hygon_qobs_daily", 
            "fn_stats": None, 
            "stations": [91000001, 15300002],},

                     }

source_dic_hourly = {

    "spw":{"coord":"spw_gauges_spw",
            "fn":"spw_qobs_hourly", #entry in datacatalog 
            "fn_stats": "spw_qstats_hourly",
            "stations": [7319, 5921, 6228, 6621, 9434, 9021, 8221],},

    "wl":{"coord":"wl_gauges_waterschaplimburg",
            "fn":"wl_qobs_hourly", 
            "fn_stats": "wl_qstats_obs_hourly",
            "stations": [1036, 1030],},

    "hp":{"coord":"hp_gauges_hydroportail",
            "fn": "hp_qobs_hourly", 
            "fn_stats": "hp_qstats_hourly",            
            "stations": [1022001001, 1720000001],},

    "rwsinfo":{"coord":"rwsinfo_gauges_rwsinfo",
            "fn":"rwsinfo_qobs_hourly", 
            "fn_stats": None,   
            "stations": [16],},    

    "france":{"coord":"france_gauges_france",
            "fn":"france_qobs_hourly", 
            "fn_stats": "hp_qstats_hourly",   
            "stations": [12220010],},    

    "hygon":{"coord":"hygon_gauges_hygon",
            "fn":None, 
            "fn_stats": "wl_qstats_obs_hourly",  #91000001 is called 24 
            "stations": [91000001, 15300002],},

                     }
                     
if dt == "daily":
        source_dic = source_dic_daily
if dt == "hourly":
        source_dic = source_dic_hourly

#%%
#Defining a color scheme for winter (10-11-12-01-02-03) VS summer (04 to 09) 
cmap = (mpl.colors.ListedColormap(['dodgerblue', 'gold', 'dodgerblue']))
bounds = [1, 4, 10, 12]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

#%%
size_fig = (10,20)
cols = [ str(n) for n in np.arange(64, 1024+64, 64)]

for case in dict_cases.keys():
        fig_tradi_gumbel, axs_tradi_gumbel = plt.subplots(ncols=2, nrows=4, figsize=size_fig, sharex=False, sharey=False) #, 
        ax_tradi_gumbel = axs_tradi_gumbel.reshape(-1)

        fig_tradi_gev, axs_tradi_gev = plt.subplots(ncols=2, nrows=4, figsize=size_fig, sharex=False, sharey=False) #, 
        ax_tradi_gev = axs_tradi_gev.reshape(-1)

        fig_ggev_1040, axs_ggev_1040 = plt.subplots(ncols=2, nrows=4, figsize=size_fig, sharex=False, sharey=False) #, 
        ax_ggev_1040 = axs_ggev_1040.reshape(-1)

        fig_ggev_65, axs_ggev_65 = plt.subplots(ncols=2, nrows=4, figsize=size_fig, sharex=False, sharey=False) #, 
        ax_ggev_65 = axs_ggev_65.reshape(-1)

        fig_shape, axs_shape= plt.subplots(ncols=2, nrows=4, figsize=size_fig, sharex=False, sharey=False) #, 
        ax_shape = axs_shape.reshape(-1)

        fig_loc, axs_loc= plt.subplots(ncols=2, nrows=4, figsize=size_fig, sharex=False, sharey=False) #, 
        ax_loc = axs_loc.reshape(-1)

        fig_scale, axs_scale= plt.subplots(ncols=2, nrows=4, figsize=size_fig, sharex=False, sharey=False) #, 
        ax_scale = axs_scale.reshape(-1)

        i=0
#        for source in source_dic:
        for station_name in dict_cases[case].keys():
                print(station_name)
                source = dict_cases[case][station_name]['source']
                station = dict_cases[case][station_name]['id']

                #Reading the data for the source
                coord_name = source_dic[source]["coord"] #coordinate in output_scalar_nc
                fn = source_dic[source]["fn"] #name of entry in datacatalog
                fn_stats = source_dic[source]["fn_stats"] #name of entry in datacatalog
                stations_sel = source_dic[source]["stations"] #selected stations 

                #get official stats
                if fn_stats != None:
                        ds_stats = mod.data_catalog.get_geodataset(fn_stats, single_var_as_array=False)
                        ds_stats = ds_stats.rename({"wflow_id":"stations"})
                else:
                        ds_stats = None

                # for station in source_dic[source]['stations']:
                #         if station not in dict_cases[case]["sel_stations_case"]:
                #                 continue
                data_Tcal = all_results[dt][source]['results'] 
                bm_summary = data_Tcal[station]['MOD']["BM_fit"]
                station_name = data_Tcal[station]['station_name']
                #station_name = 'test'

                max_y = roundup(max(bm_summary[str(1024)]['return_levels']['gumbel_r'].max(), bm_summary[str(1024)]['return_levels']['genextreme'].max(), data_Tcal[station]['MOD']['emp_T'].value.max(), bm_summary[str(1024)]['conf_int_high']['gumbel_r'].max(), bm_summary[str(1024)]['conf_int_high']['genextreme'].max()))
                distr='gumbel_r'
                plot_return_period_tradi(ax_tradi_gumbel[i], cmap, norm, data_Tcal, ds_stats, [0, max_y], [1,2000], station, distr, station_name) #station_name

                distr='genextreme'
                # f, ax = plt.subplots()
                plot_return_period_tradi(ax_tradi_gev[i], cmap, norm, data_Tcal, ds_stats, [0, max_y], [1,2000], station, distr, station_name)

                #All years - 1040 years
                n=1024
                plot_gumbel_gev_n(ax_ggev_1040[i], cmap, norm, data_Tcal, ds_stats, n, [0, max_y], [1,2000], station, f"{station_name} - {n} years")

                #One ensemble - 65 years
                n=64
                plot_gumbel_gev_n(ax_ggev_65[i], cmap, norm, data_Tcal, ds_stats, n, [0, max_y], [1,2000], station, f"{station_name} - {n} years")
        
                #Plot convergence
                plot_convergence_params_type(ax_shape[i], data_Tcal, station, cols, distr, 'shape', f"{station_name} - {distr}")
                plot_convergence_params_type(ax_loc[i], data_Tcal, station, cols, distr, 'loc', f"{station_name} - {distr}")
                plot_convergence_params_type(ax_scale[i], data_Tcal, station, cols, distr, 'scale', f"{station_name} - {distr}")
                
                i+=1
                print(f'Station {station} done!')
        # plt.show()

        if (i == 7) and (case =='report_appendix'):
                ax_tradi_gumbel[7].set_visible(False)
                ax_tradi_gev[7].set_visible(False)
                ax_ggev_1040[7].set_visible(False)
                ax_ggev_65[7].set_visible(False)
                ax_shape[7].set_visible(False)
                ax_loc[7].set_visible(False)
                ax_scale[7].set_visible(False)


        fig_tradi_gumbel.savefig(os.path.join(fn_fig, f'{case}_{dt}_gumbel_r_return_curve_tradi.png'), dpi=400)
        print('Figure tradi gumbel saved!')
        fig_tradi_gev.savefig(os.path.join(fn_fig, f'{case}_{dt}_genextreme_return_curve_tradi.png'), dpi=400)
        print('Figure tradi gev saved!')
        fig_ggev_1040.savefig(os.path.join(fn_fig, f'{case}_{dt}_gumbel_gev_return_curve_1024.png'), dpi=400)
        print('Figure gumbel gev 1040 saved!')
        fig_ggev_65.savefig(os.path.join(fn_fig, f'{case}_{dt}_gumbel_gev_return_curve_64.png'), dpi=400)
        print('Figure gumbel gev 65 saved!')
        fig_shape.savefig(os.path.join(fn_fig, f'{case}_{dt}_shape_conv.png'), dpi=400)
        print('Figure gumbel gev 65 saved!')
        fig_loc.savefig(os.path.join(fn_fig, f'{case}_{dt}_location_conv.png'), dpi=400)
        print('Figure gumbel gev 65 saved!')
        fig_scale.savefig(os.path.join(fn_fig, f'{case}_{dt}_scale_conv.png'), dpi=400)
        print('Figure gumbel gev 65 saved!')
        print('Done!')
# %%
