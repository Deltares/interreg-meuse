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
model_wflow = "p_geulrur" #"o_rwsinfo"
Folder_p = os.path.join(Folder_start, "wflow", model_wflow)
folder = 'members_bias_corrected_revised_daily' #"members_bias_corrected_revised_hourly"#"members_bias_corrected_revised_daily" #"members_bias_corrected_hourly"
fn_fig = os.path.join(Folder_start, "Figures", model_wflow, folder)
fn_data_out = os.path.join(fn_fig, 'data')

dt = folder.split("_")[-1]
print(f"Performing analysis for {dt}")

if not os.path.exists(fn_fig):
    os.makedirs(fn_fig)

#We import the dictionary with the results
fn_out = os.path.join(fn_data_out, f'{dt}_results_stations.pickle')
with open(fn_out, 'rb') as f:
    daily_results = pickle.load(f)

#%%
folder = 'members_bias_corrected_revised_hourly' #"members_bias_corrected_revised_hourly"#"members_bias_corrected_revised_daily" #"members_bias_corrected_hourly"
fn_fig = os.path.join(Folder_start, "Figures", model_wflow, folder)
fn_data_out = os.path.join(fn_fig, 'data')

dt = folder.split("_")[-1]
print(f"Performing analysis for {dt}")

if not os.path.exists(fn_fig):
    os.makedirs(fn_fig)

#We import the dictionary with the results
fn_out = os.path.join(fn_data_out, f'{dt}_results_stations.pickle')
with open(fn_out, 'rb') as f:
    hourly_results = pickle.load(f)

#%% We define the stations we want to show
# dict_cases = {
#         "report_main":  
#                 {"sel_stations_case":[7319, 5921, 6228, 1036, 1022001001, 1720000001, 16, 91000001],},
#         "report_appendix": 
#                 {"sel_stations_case":[6621, 9434, 9021, 8221, 1030, 12220010, 15300002]}
# }
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
                 'Vesdre at Chaudfontaine': {'lat': 50.59167, 'lon': 5.65250, 'id': 6228, 'source': 'spw'},
                 'Semois at Membre Pont': {'lat': 49.86667, 'lon': 4.90250, 'id': 9434, 'source': 'spw'},
                 'Lesse at Gendron': {'lat': 50.20833, 'lon': 4.96083, 'id': 8221, 'source': 'spw'},
                 'Rur at Stah': {'lat': 51.1, 'lon': 6.10250, 'id': 91000001, 'source': 'hygon'},
                 'Meuse at St-Mihiel': {'lat': 48.86667, 'lon': 5.52750, 'id': 12220010, 'source': 'france'},},
}
#%%
def plot_fig_dt_T(ax, daily_results, hourly_results, ylims, xlims, station, title):
    df = daily_results['daily'][source]['results'][station]["MOD"]['emp_T'].sort_values('emp_rp')
    ax.plot(df.emp_rp.values, df.value, c = 'b', marker='.', lw = 1, label='daily')
    
    df = hourly_results['hourly'][source]['results'][station]["MOD"]['emp_T'].sort_values('emp_rp')
    ax.plot(df.emp_rp.values, df.value, c = 'r', marker='*', lw = 1, label='hourly')
    
    ax.set_xscale('log')
    ax.set_ylim(ylims)
    ax.set_xlim(xlims)
    ax.set_title(title)
    ax.legend()
    ax.set_xlabel('Return Period (year)')
    ax.set_ylabel('Discharge (m3/s)')    

def plot_fig_dt_T_diff(ax, daily_results, hourly_results, ylims, xlims, station, title):
    df_daily = daily_results['daily'][source]['results'][station]["MOD"]['emp_T'].sort_values('emp_rp').reset_index(drop=True)
    df_hourly = hourly_results['hourly'][source]['results'][station]["MOD"]['emp_T'].sort_values('emp_rp').reset_index(drop=True)
    df_diff = df_hourly - df_daily
    ax.plot(df_hourly.emp_rp.values, df_diff.value, c = 'k', marker='.')
    ax.set_xscale('log')
    #ax.set_ylim(ylims)
    ax.set_xlim(xlims)
    ax.set_title(title)
    ax.set_xlabel('Return Period (year)')
    ax.set_ylabel('Diff. in Discharge (m3/s)')   

def plot_fig_dt_T_diff_perc(ax, daily_results, hourly_results, ylims, xlims, station, title):
    df_daily = daily_results['daily'][source]['results'][station]["MOD"]['emp_T'].sort_values('emp_rp').reset_index(drop=True)
    df_hourly = hourly_results['hourly'][source]['results'][station]["MOD"]['emp_T'].sort_values('emp_rp').reset_index(drop=True)
    df_diff = df_hourly - df_daily
    df_perc = (df_diff.value*100)/df_daily.value
    ax.plot(df_hourly.emp_rp.values, df_perc, c = 'k', marker='.')
    ax.set_xscale('log')
    max_y = max(100,np.round(df_perc.iloc[104:].max()+1))
    ax.set_ylim([0, max_y])
    ax.set_xlim(xlims)
    ax.set_title(title)
    ax.set_xlabel('Return Period (year)')
    ax.set_ylabel('Perc. diff w.r.t. to daily')   

def clean_subaxis(axs):
    for ax in axs.reshape(-1):
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title('')

#%% We plot the difference 
size_fig = (10,20)
for case in dict_cases.keys():
    print(case)
    fig_dt, axs_dt = plt.subplots(ncols=2, nrows=4, figsize=size_fig, sharex=False, sharey=False) #, 
    ax_dt = axs_dt.reshape(-1)

    fig_diff, axs_diff = plt.subplots(ncols=2, nrows=4, figsize=size_fig, sharex=False, sharey=False) #, 
    ax_diff = axs_diff.reshape(-1)

    fig_diff_perc, axs_diff_perc = plt.subplots(ncols=2, nrows=4, figsize=size_fig, sharex=False, sharey=False) #, 
    ax_diff_perc = axs_diff_perc.reshape(-1)

    fig_all, axs_all = plt.subplots(ncols=3, nrows=8, figsize=size_fig, sharex=False, sharey=False) #, 

    j=0
    i=0
    for station_name in dict_cases[case].keys():
        source = dict_cases[case][station_name]['source']
        station = dict_cases[case][station_name]['id']

        max_y = roundup(max(daily_results['daily'][source]['results'][station]["MOD"]['emp_T'].value.max(), hourly_results['hourly'][source]['results'][station]["MOD"]['emp_T'].value.max()))
        #max_y = 2000
        plot_fig_dt_T(ax_dt[i], daily_results, hourly_results, [0, max_y], [1,2000], station, station_name)
        plot_fig_dt_T_diff(ax_diff[i], daily_results, hourly_results, [0, max_y], [1,2000], station, station_name)
        plot_fig_dt_T_diff_perc(ax_diff_perc[i], daily_results, hourly_results, [0, max_y], [1,2000], station, station_name)
        
        plot_fig_dt_T(axs_all[i,0], daily_results, hourly_results, [0, max_y], [1,2000], station, station_name)
        plot_fig_dt_T_diff(axs_all[i,1], daily_results, hourly_results, [0, max_y], [1,2000], station, station_name)
        plot_fig_dt_T_diff_perc(axs_all[i,2], daily_results, hourly_results, [0, max_y], [1,2000], station, station_name)
        
        clean_subaxis(axs_all[i,:])
        axs_all[i,0].set_ylabel(station_name)
        i += 1 

    axs_all[0,0].set_title('Discharge (m3/s)')
    axs_all[0,1].set_title('Diff in discharge (m3/s)')
    axs_all[0,2].set_title('Perc. diff with daily')

    if i == 7:
        [ax.set_visible(False) for ax in axs_all[7,:].reshape(-1)]
        ax_dt[7].set_visible(False)
        ax_diff[7].set_visible(False)
        ax_diff_perc[7].set_visible(False)
        
    fig_dt.savefig(os.path.join(fn_fig, f'{case}_dt_emp_return_curve_1040.png'), dpi=400)
    print('Figure return periods dt saved!')
    print(os.path.join(fn_fig, f'{case}_dt_emp_return_curve_1040.png'))
    fig_diff.savefig(os.path.join(fn_fig, f'{case}_dt_emp_return_curve_diff_1040.png'), dpi=400)
    print('Figure return periods dt diff saved!!')
    fig_diff_perc.savefig(os.path.join(fn_fig, f'{case}_dt_emp_return_curve_diff_perc_1040.png'), dpi=400)
    print('Figure return periods dt diff perc saved!!')
    fig_all.savefig(os.path.join(fn_fig, f'{case}_dt_emp_return_curve_ALL.png'), dpi=400)
    print('Done!')


# %%
