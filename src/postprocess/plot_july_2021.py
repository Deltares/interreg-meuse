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
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.dates import (AutoDateLocator, YearLocator, MonthLocator,
                              DayLocator, WeekdayLocator, HourLocator,
                              MinuteLocator, SecondLocator, MicrosecondLocator,
                              RRuleLocator, rrulewrapper, MONTHLY,
                              MO, TU, WE, TH, FR, SA, SU, DateFormatter,
                              AutoDateFormatter, ConciseDateFormatter)

#%% We import the modelled data
Folder_start = r"/p/11208719-interreg"
model_wflow = "p_geulrur"#"o_rwsinfo"
Folder_p = os.path.join(Folder_start, "wflow", model_wflow)
folder = 'members_bias_corrected_revised_daily' #"members_bias_corrected_revised_hourly"#"members_bias_corrected_revised_daily" #"members_bias_corrected_hourly"
#folder = sys.argv[1] 
fn_fig = os.path.join(Folder_start, "Figures", model_wflow, folder)
fn_data_out = os.path.join(fn_fig, 'data')

dt = folder.split("_")[-1]
print(f"Performing analysis for {dt}")

if not os.path.exists(fn_fig):
    os.makedirs(fn_fig)

# #We import the dictionary with the results
# fn_out = os.path.join(fn_data_out, f'{dt}_results_stations.pickle')
# with open(fn_out, 'rb') as f:
#     all_results = pickle.load(f)


#%% read one model 
root = r"/p/11208719-interreg/wflow/p_geulrur"#r"/p/11208719-interreg/wflow/o_rwsinfo"
config_fn = f"run_geulrur\run_geulrur.toml" #f"{folder}/r11i1p5f1/{folder}_r11i1p5f1.toml"
yml = r"/p/11208719-interreg/data/data_meuse.yml" #r"/p/11208719-interreg/data/data_meuse_linux.yml"
mod = WflowModel(root = root, config_fn=config_fn, data_libs=["deltares_data", yml], mode = "r")

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
                 'Ambleve at Martinrive': {'lat': 50.48333, 'lon': 5.63583, 'id': 6621, 'source': 'spw'}, #or this one?
                 'Semois at Membre Pont': {'lat': 49.86667, 'lon': 4.90250, 'id': 9434, 'source': 'spw'}, # This one?
                 'Lesse at Gendron': {'lat': 50.20833, 'lon': 4.96083, 'id': 8221, 'source': 'spw'},
                 'Rur at Stah': {'lat': 51.1, 'lon': 6.10250, 'id': 91000001, 'source': 'hygon'},
                 'Meuse at St-Mihiel': {'lat': 48.86667, 'lon': 5.52750, 'id': 12220010, 'source': 'france'},},
}
#%% make dic of all sources 
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
            "fn_stats": None,   
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
            "fn_stats": None,   
            "stations": [12220010],},    

    "hygon":{"coord":"hygon_gauges_hygon",
            "fn":None, 
            "fn_stats": None, 
            "stations": [91000001, 15300002],},

                     }

#%% Storing the time series
# fn_out_src = r'p:\11208719-interreg\data\July2021_csv\a_raw'
# var = "Q"
# for dt in ["hourly", "daily"]:
#         print(dt)
#         if dt == 'daily':
#                 source_dic = source_dic_daily
#         if dt == 'hourly':
#                 source_dic = source_dic_hourly
        
#         for source in source_dic.keys():
#                 print(source)
#                 #Reading the data for the source
#                 coord_name = source_dic[source]["coord"] #coordinate in output_scalar_nc
#                 fn = source_dic[source]["fn"] #name of entry in datacatalog
#                 #fn_stats = source_dic[source]["fn_stats"] #name of entry in datacatalog
#                 stations_sel = source_dic[source]["stations"] #selected stations 
                
#                 #get observed ds
#                 if fn != None:
#                         print(f"Opening {fn}")
#                         ds_obs = mod.data_catalog.get_geodataset(fn, variables = ["Q"])
#                         ds_obs_sel = ds_obs.sel(wflow_id = stations_sel)
#                         #rename wflow_id to stations
#                         ds_obs_sel = ds_obs_sel.rename({"wflow_id":"stations"})

#                         for station in ds_obs_sel.stations.values:
#                                 print(station)
#                                 ts_key = ds_obs_sel.sel(stations=station).to_series()
#                                 first_idx = ts_key.first_valid_index()
#                                 last_idx = ts_key.last_valid_index()
#                                 #print(f'Station {station_name}, {dt}, data start: {first_idx}')
#                                 #print(f'Station {station_name}, {dt}, data start: {last_idx}')
#                                 ts_key = ts_key.loc[first_idx:last_idx]

#                                 #We select for the July 2021 event
#                                 df = ts_key.loc['2021-07-01':'2021-08-01']
#                                 fn_out_fn = os.path.join(fn_out_src, f"{dt}", f"{source}")
#                                 if not os.path.exists(fn_out_fn):
#                                         os.makedirs(fn_out_fn)
#                                 df.to_csv(os.path.join(fn_out_fn, f"{station}.csv"), index_label='time')
#                                 print('Done!')

# %% Reading the 
#size_fig = (10,20)
date_parser_daily = lambda x: datetime.strptime(x, "%Y-%m-%d")
date_parser_hourly = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
# case = "report_main"
fn_in_src = r'p:\11208719-interreg\data\July2021_csv\a_raw'
for case in dict_cases.keys():
        print(case)
        fig, axs = plt.subplots(ncols=2, nrows=4, sharex=False, sharey=False) #, figsize=size_fig, 
        ax = axs.reshape(-1)
        i=0
        for station_name in dict_cases[case].keys():
                print(station_name)
                source = dict_cases[case][station_name]['source']
                station = dict_cases[case][station_name]['id']      

                #We load the final time series
                if os.path.exists(os.path.join(fn_in_src, 'hourly', f'{source}', f'{station}.csv')):
                        df_hourly = pd.read_csv(os.path.join(fn_in_src, 'hourly', f'{source}', f'{station}.csv'),
                                                index_col=['time'], parse_dates=['time'], date_parser = date_parser_hourly)
                else:
                        df_hourly = pd.DataFrame(data= None,columns=['time', 'Q']).set_index('time')     

                if os.path.exists(os.path.join(fn_in_src, 'daily', f'{source}', f'{station}.csv')):                
                        df_daily = pd.read_csv(os.path.join(fn_in_src, 'daily', f'{source}', f'{station}.csv'),
                                                index_col=['time'], parse_dates=['time'], date_parser = date_parser_daily)
                else:
                        df_daily = pd.DataFrame(data= None,columns=['time', 'Q']).set_index('time')   

                ax[i].plot(df_hourly.index, df_hourly.values, marker = 'o', ls='-', lw=0.5, c='b', label = 'hourly') #df.plot(ax=ax)
                ax[i].plot(df_daily.index, df_daily.values, marker = 'o', ls='-', lw=0.5, c='r', label = 'daily') 
                ax[i].set_ylabel('Discharge ($m^3/s$)')
                ax[i].set_ylim(bottom=0)
                ax[i].set_title(station_name)
                ax[i].legend()      
                i+=1

#%% Correcting manually the data 
# dt = 'daily'
# source = 'hygon'
# station = 91000001
# fn_in_src = r'p:\11208719-interreg\data\July2021_csv\a_raw'
# fn_out_src = r'p:\11208719-interreg\data\July2021_csv\b_modif'
# #%%
# df_hourly = pd.read_csv(os.path.join(fn_in_src, 'hourly', f'{source}', f'{station}.csv'), index_col=['time'], parse_dates=['time'], date_parser = date_parser_hourly)
# df_hourly['note'] = 'invalid'

# #%%
# df_daily = pd.read_csv(os.path.join(fn_in_src, 'daily', f'{source}', f'{station}.csv'), index_col=['time'], parse_dates=['time'], date_parser = date_parser_daily)
# df_daily['note'] = 'invalid'
# #df_daily = pd.read_csv(os.path.join(fn_in_src, 'daily', f'{source}', f'{station}.csv'), index_col=['time'], parse_dates=['time'], date_parser = date_parser_daily)

# #%%                
# df_daily = df_hourly.resample("D").mean()

# #%%
# df_daily.to_csv(os.path.join(fn_out_src,f"{dt}", f"{source}", f"{station}.csv"), index_label='time', date_format = "%Y-%m-%d")

#%% We add the modified data collected - we do this manually 
fn_in_src = r'p:\11208719-interreg\data\July2021_csv\a_raw'
fn_out_src = r'p:\11208719-interreg\data\July2021_csv\b_modif'
var = "Q"
date_parser_daily = lambda x: datetime.strptime(x, "%Y-%m-%d")
date_parser_hourly = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")

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
                 'Ambleve at Martinrive': {'lat': 50.48333, 'lon': 5.63583, 'id': 6621, 'source': 'spw'}, #or this one?
                 'Semois at Membre Pont': {'lat': 49.86667, 'lon': 4.90250, 'id': 9434, 'source': 'spw'}, # This one?
                 'Lesse at Gendron': {'lat': 50.20833, 'lon': 4.96083, 'id': 8221, 'source': 'spw'},
                 'Rur at Stah': {'lat': 51.1, 'lon': 6.10250, 'id': 91000001, 'source': 'hygon'},
                 'Meuse at St-Mihiel': {'lat': 48.86667, 'lon': 5.52750, 'id': 12220010, 'source': 'france'},},
}

#%% We create the final time series and save them      
# values_2021_daily = pd.DataFrame()
# values_2021_hourly = pd.DataFrame()  
# fn_in_raw_src = r'p:\11208719-interreg\data\July2021_csv\a_raw'  
# fn_in_modif_src = r'p:\11208719-interreg\data\July2021_csv\b_modif' 
# fn_out_src = r'p:\11208719-interreg\data\July2021_csv\c_final'
# for case in dict_cases.keys():
#         print(case)
#         for station_name in dict_cases[case].keys():
#                 print(station_name)
#                 source = dict_cases[case][station_name]['source']
#                 station = dict_cases[case][station_name]['id']
#                 print(source)
#                 print(station)

#                 #We read the hourly values - raw
#                 if os.path.exists(os.path.join(fn_in_raw_src, 'hourly', f'{source}', f'{station}.csv')):
#                         df_hourly = pd.read_csv(os.path.join(fn_in_raw_src, 'hourly', f'{source}', f'{station}.csv'),
#                                                 index_col=['time'], parse_dates=['time'], date_parser = date_parser_hourly)
#                         df_hourly['note'] = 'raw'
#                 else:
#                         df_hourly = pd.DataFrame(data= None,columns=['time', 'Q']).set_index('time')   
#                 df_hourly = df_hourly.dropna(how='any')  
                
#                 #We read the modified values
#                 #We read the hourly values - raw
#                 if os.path.exists(os.path.join(fn_in_modif_src, 'hourly', f'{source}', f'{station}.csv')):
#                         df_hourly2 = pd.read_csv(os.path.join(fn_in_modif_src, 'hourly', f'{source}', f'{station}.csv'),
#                                                 index_col=['time'], parse_dates=['time'], date_parser = date_parser_hourly)
#                 else:
#                         df_hourly2 = pd.DataFrame(data= None,columns=['time', 'Q']).set_index('time')  
                
#                 #We store the final data
#                 df_hourly_final = pd.concat([df_hourly, df_hourly2], axis = 0).sort_index(ascending=True)
#                 fn_out_fn = os.path.join(fn_out_src, "hourly", f"{source}")
#                 if not os.path.exists(fn_out_fn):
#                         os.makedirs(fn_out_fn)                
#                 df_hourly_final.to_csv(os.path.join(fn_out_fn, f"{station}.csv"), index_label='time', date_format = "%Y-%m-%d %H:%M:%S")

#                 #We read the daily values - raw
#                 if os.path.exists(os.path.join(fn_in_raw_src, 'daily', f'{source}', f'{station}.csv')):
#                         df_daily = pd.read_csv(os.path.join(fn_in_raw_src, 'daily', f'{source}', f'{station}.csv'),
#                                                 index_col=['time'], parse_dates=['time'], date_parser = date_parser_daily)
#                         df_daily['note'] = 'raw'
#                 else:
#                         df_daily = pd.DataFrame(data= None,columns=['time', 'Q']).set_index('time')   
#                 df_daily = df_daily.dropna(how='any')  
                
#                 #We read the modified values
#                 if os.path.exists(os.path.join(fn_in_modif_src, 'daily', f'{source}', f'{station}.csv')):
#                         df_daily2 = pd.read_csv(os.path.join(fn_in_modif_src, 'daily', f'{source}', f'{station}.csv'),
#                                                 index_col=['time'], parse_dates=['time'], date_parser = date_parser_daily)
#                 else:
#                         df_daily2 = pd.DataFrame(data= None,columns=['time', 'Q']).set_index('time')  
                
#                 #We store the final data
#                 df_daily_final = pd.concat([df_daily, df_daily2], axis = 0).sort_index(ascending=True)

#                 fn_out_fn = os.path.join(fn_out_src, "daily", f"{source}")
#                 if not os.path.exists(fn_out_fn):
#                         os.makedirs(fn_out_fn)                
#                 df_daily_final.to_csv(os.path.join(fn_out_fn, f"{station}.csv"), index_label='time', date_format = "%Y-%m-%d")



# %% Plotting the new data

size_fig = (10,20)
date_parser_daily = lambda x: datetime.strptime(x, "%Y-%m-%d")
date_parser_hourly = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
# case = "report_main"
fn_in_src = '/p/11208719-interreg/data/July2021_csv/c_final'
df_daily_max = pd.DataFrame(columns=['max'])
df_hourly_max = pd.DataFrame(columns=['max'])
for case in dict_cases.keys():
        print(case)
        fig, axs = plt.subplots(ncols=2, nrows=4, sharex=False, sharey=False, figsize=size_fig) #, figsize=size_fig, 
        ax = axs.reshape(-1)
        i=0
        for station_name in dict_cases[case].keys():
                print(station_name)
                source = dict_cases[case][station_name]['source']
                station = dict_cases[case][station_name]['id']      

                #We load the final time series
                if os.path.exists(os.path.join(fn_in_src, 'hourly', f'{source}', f'{station}.csv')):
                        df_hourly = pd.read_csv(os.path.join(fn_in_src, 'hourly', f'{source}', f'{station}.csv'),
                                                index_col=['time'], parse_dates=['time'], date_parser = date_parser_hourly)
                else:
                        df_hourly = pd.DataFrame(data= None,columns=['time', 'Q']).set_index('time')     

                if os.path.exists(os.path.join(fn_in_src, 'daily', f'{source}', f'{station}.csv')):                
                        df_daily = pd.read_csv(os.path.join(fn_in_src, 'daily', f'{source}', f'{station}.csv'),
                                                index_col=['time'], parse_dates=['time'], date_parser = date_parser_daily)
                else:
                        df_daily = pd.DataFrame(data= None,columns=['time', 'Q']).set_index('time')   

                ax[i].plot(df_hourly.index, df_hourly['Q'].values, marker = 'o', ls='-', lw=0.5, c='b', label = 'hourly') #df.plot(ax=ax)

                other_hourly = False
                #We separate the 
                if ('note' in df_hourly.columns):
                        if ('estimated' in df_hourly['note'].values):
                                df_hourly_other = df_hourly.where(df_hourly['note'] == 'estimated').dropna(how='all')
                                df_hourly.drop(df_hourly[df_hourly['note'] == 'estimated'].index, inplace = True)
                                other_hourly = True
                
                ax[i].plot(df_daily.index, df_daily['Q'].values, marker = 'o', ls='-', lw=0.5, c='r', label = 'daily') 

                if other_hourly == True:
                        ax[i].plot(df_hourly_other.index, df_hourly_other['Q'].values, marker = 'o', ls='-', lw=0.5, c='cyan', label = 'hourly estimated') #df.plot(ax=ax)

                ax[i].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=SU, interval=1))
                ax[i].xaxis.set_minor_locator(mdates.DayLocator())
                # Text in the x-axis will be displayed in 'YYYY-mm' format.
                ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
                # Rotates and right-aligns the x labels so they don't crowd each other.
                for label in ax[i].get_xticklabels(which='major'):
                        label.set(rotation=30, horizontalalignment='right')
                ax[i].set_ylabel('Discharge ($m^3/s$)')
                ax[i].set_ylim(bottom=0)
                ax[i].set_title(station_name)
                ax[i].legend()  
                i+=1

                df_daily_max.loc[station,'max'] = df_daily['Q'].max()
                df_hourly_max.loc[station,'max'] = df_hourly['Q'].max()
        fig.savefig(os.path.join(fn_fig, f'{case}_adapted_july_2021_final.png'), dpi=400)
#df_daily_max.to_csv('/p/11208719-interreg/data/July2021_csv/c_final/July2021_daily_max.csv',index_label='station')
#df_hourly_max.to_csv('/p/11208719-interreg/data/July2021_csv/c_final/July2021_hourly_max.csv',index_label='station')
# %%
