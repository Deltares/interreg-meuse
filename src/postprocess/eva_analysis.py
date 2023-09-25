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
import os, glob, sys
from scipy.stats import gumbel_r, genextreme
import xarray as xr
import math
from matplotlib.pyplot import cm
import hydromt
from hydromt_wflow import WflowModel
import geopandas as gpd
import pickle
from eva_analysis_functions import * 
import random

#%% We import the modelled data
Folder_start = "/p/11208719-interreg"
model_wflow = "p_geulrur" #"o_rwsinfo"
Folder_p = os.path.join(Folder_start, "wflow", model_wflow)

#folder = 'members_bias_corrected_revised_daily' #"members_bias_corrected_revised_hourly"#"members_bias_corrected_revised_daily" #"members_bias_corrected_hourly"
folder = sys.argv[1] 

fn_fig = os.path.join(Folder_start, "Figures", model_wflow, folder)
fn_data_out = os.path.join(fn_fig, 'data')

dt = folder.split("_")[-1]
print(f"Performing analysis for {dt}")

if not os.path.exists(fn_fig):
    os.makedirs(fn_fig)

if not os.path.exists(fn_data_out):
    os.makedirs(fn_data_out)
#%% read one model 
root = r"/p/11208719-interreg/wflow/p_geulrur" #r"/p/11208719-interreg/wflow/o_rwsinfo"
config_fn = f"{folder}/r11i1p5f1/{folder}_r11i1p5f1.toml"
yml = r"/p/11208719-interreg/data/data_meuse_linux.yml"
mod = WflowModel(root = root, config_fn=config_fn, data_libs=["deltares_data", yml], mode = "r")
#%%
dict_station_names = {
        7319: 'Sambre at Salzinne',
        5921: 'Ourthe at Tabreux',
        6228: 'Vesdre at Chaudfontaine',
        6621: 'Ambleve at Martinrive',
        9434: 'Semois at Membre Pont',
        9021: 'Viroin Treignes',
        8221: 'Lesse at Gendron',
        1036: 'Geul at Meerssen',
        1030: 'Geul at Hommerich',
        1022001001: 'Meuse at Goncourt',
        1720000001: 'Meuse at Chooz',
        16: 'Meuse at St Pieter',
        12220010: 'Meuse at St-Mihiel',
        91000001: 'Rur at Stah',
        15300002: 'Rur at Monschau'}

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

runs_dict = {}
for key in model_runs.keys():
    print(key)
    case = model_runs[key]["case"]
    folder = model_runs[key]["folder"] 
    # runs_dict[key] = pd.read_csv(os.path.join(Folder_p, case, folder, "output.csv"), index_col=0, header=0, parse_dates=True)
    runs_dict[key] = xr.open_dataset(os.path.join(Folder_p, case, folder, "output_scalar.nc"))

#%%
#Defining the color scheme per month
# c_month = plt.cm.hsv(np.linspace(0,1,12))
# cmap = colors.LinearSegmentedColormap.from_list("months", c_month)
# norm = colors.Normalize(vmin=1, vmax=12)
#Defining a color scheme for winter (10-11-12-01-02-03) VS summer (04 to 09) 
cmap = (mpl.colors.ListedColormap(['dodgerblue', 'gold', 'dodgerblue']))
bounds = [1, 4, 10, 12]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
#%%
all_results = dict()
all_results[dt] = dict()

var = "Q"
for source in source_dic:
        print(source)
        #%%              
        all_results[dt][source] = dict()
        #Reading the data for the source
        coord_name = source_dic[source]["coord"] #coordinate in output_scalar_nc
        fn = source_dic[source]["fn"] #name of entry in datacatalog
        fn_stats = source_dic[source]["fn_stats"] #name of entry in datacatalog
        stations_sel = source_dic[source]["stations"] #selected stations 

        #get observed ds
        if fn != None:
                ds_obs = mod.data_catalog.get_geodataset(fn, variables = ["Q"])
                ds_obs_sel = ds_obs.sel(wflow_id = stations_sel)
                #add coordinate run for plots laurene 
                ds_obs_sel = ds_obs_sel.assign_coords({"runs":"Obs."}).expand_dims("runs")
                #rename wflow_id to stations
                ds_obs_sel = ds_obs_sel.rename({"wflow_id":"stations"})

        #get official stats
        if fn_stats != None:
                ds_stats = mod.data_catalog.get_geodataset(fn_stats, single_var_as_array=False)
                ds_stats = ds_stats.rename({"wflow_id":"stations"})
        else:
                print(f"No official stats for {source}")
                ds_stats = None

        ds_mods = []
        for key in runs_dict.keys(): 
                print(key)
                ds_mod = runs_dict[key]
                #ds_mod_sel = ds_mod.sel({f"{var}_{coord_name}" : list(map(str, stations_sel))}).sel(time = slice(start, end))
                ds_mod_sel = ds_mod.sel({f"{var}_{coord_name}" : list(map(str, stations_sel))})
                #rename Q_source to Q 
                ds_mod_sel = ds_mod_sel.rename({f"{var}_{source}":f"{var}"})[f"{var}"]
                #chunk 
                if dt == "daily":
                        ds_mod_sel = ds_mod_sel.chunk({f"{var}_{coord_name}":len(stations_sel)}) #  (dict(Q_spw_gauges_spw=3))
                if dt == "hourly":
                        ds_mod_sel = ds_mod_sel.chunk({f"{var}_{coord_name}":1}) #  (dict(Q_spw_gauges_spw=3))
                #add runs coord 
                ds_mod_sel = ds_mod_sel.assign_coords({"runs":key}).expand_dims("runs")
                #rename coord to "station"
                ds_mod_sel = ds_mod_sel.rename({f"{var}_{coord_name}":"stations"})
                #make sure stations is int instead of str
                ds_mod_sel["stations"] = list(map(int, list(ds_mod_sel["stations"].values)))
                ds_mods.append(ds_mod_sel) 

        #combine obs and runs in one dataset 
        ds_mod_sel = xr.concat(ds_mods, dim='runs')
        if fn != None:
                ds = xr.concat([ds_obs_sel, ds_mod_sel], dim = "runs").to_dataset()
        else:
                ds = ds_mod_sel.to_dataset()
        #%%        
        #ds = ds.load()

        #%% We plot the observed data and perform statistics there
        data_Tcal = dict()
        for station in ds.stations.values:
                print(station)
                #%%
                data_Tcal[station] = dict()
                data_Tcal[station]['OBS'] = dict()

                Ts = [1.1, 1.5, 1.8, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 1200, 2000]
                summ = pd.DataFrame(columns=['genextreme', 'gumbel_r', 'expon', 'genpareto'], index=['loc', 'scale', 'shape', 'AIC', 'rate']) 
                ret_levels = pd.DataFrame(columns=['genextreme', 'gumbel_r', 'expon', 'genpareto'], index=Ts) 
                cils = pd.DataFrame(columns=['genextreme', 'gumbel_r', 'expon', 'genpareto'], index=Ts) 
                cihs = pd.DataFrame(columns=['genextreme', 'gumbel_r', 'expon', 'genpareto'], index=Ts) 

                try:
                        max_2021 = pd.read_csv(f'/p/11208719-interreg/data/July2021_csv/c_final/July2021_{dt}_max.csv', index_col='station')
                        max_2021 = max_2021.loc[station, 'max']
                except:
                        max_2021 = np.nan                
                
                data_Tcal[station]['OBS']['max_2021'] = max_2021

                if 'Obs.' not in ds.runs.values:
                        print(f'No observed time series for {station}')
                        data_Tcal[station]['OBS']['peaks'] = pd.DataFrame()
                        data_Tcal[station]['OBS']['emp_T'] = pd.DataFrame(columns=['emp_rp', 'value'])
                        data_Tcal[station]['OBS']['emp_month'] = pd.DataFrame()
                        data_Tcal[station]['OBS']['BM_fit'] = {'eva_params': summ, 'return_levels': ret_levels, 'conf_int_low': cils, 'conf_int_high': cihs}
                        continue

                ts_key = ds.sel(runs='Obs.', stations=station)['Q'].to_series()
                print(f"Performing statistics on observed ts for {station}")

                #For now we select one and extract the extremes
                ts_peaks=pd.DataFrame()
                # ts_key = data_t[station] #We need to remove years with full nans!!
                first_idx = ts_key.first_valid_index()
                last_idx = ts_key.last_valid_index()
                ts_key = ts_key.loc[first_idx:last_idx]

                ts_peaks_value = ts_key.resample('AS-Oct').max()
                ts_peaks_dates = ts_key.resample('AS-Oct').agg(lambda x : np.nan if x.count() == 0 else x.idxmax())
                ts_peaks = ts_key.loc[ts_peaks_dates.values]

                #We perform the EVA - BM - Gumbel 
                # peaks = pyex.get_extremes(ts_key, method = "BM", extremes_type="high", block_size="365.2426D", errors="ignore", min_last_block=0.75)
                # ts_peaks = peaks.copy()

                emp_T = calculate_emp(ts_peaks.values)
                emp_month = ts_peaks.index.month

                data_Tcal[station]['OBS']['peaks'] = ts_peaks
                data_Tcal[station]['OBS']['emp_T'] = emp_T
                data_Tcal[station]['OBS']['emp_month'] = emp_month

                model = pyex.EVA.from_extremes(extremes = ts_peaks, method = "BM", extremes_type = 'high', block_size = "365.2425D")

                summ[['gumbel_r']], ret_levels[['gumbel_r']], cils[['gumbel_r']], cihs[['gumbel_r']] = calc_stats(model, Ts, distr="gumbel_r")
                summ[['genextreme']], ret_levels[['genextreme']], cils[['genextreme']], cihs[['genextreme']] = calc_stats(model, Ts, distr="genextreme")

                data_Tcal[station]['OBS']['BM_fit'] = {'eva_params': summ, 'return_levels': ret_levels, 'conf_int_low': cils, 'conf_int_high': cihs}

                # try:
                #         max_2021 = ts_key.loc['2021-06-01':'2021-08-01'].max()
                # except:
                #         max_2021 = np.nan
                #We load the max values and store them
                
                #We plot the time series and show the 2021 values
                station_name = dict_station_names[station]
                f, ax = plt.subplots()
                ts_key.dropna().plot(ax=ax, lw=0.5)
                ax.set_ylabel('Discharge ($m^3/s$)')
                ax.set_ylim(bottom=0)
                ax.hlines(y=max_2021, xmin=ts_key.dropna().index[0], xmax=ts_key.dropna().index[-1], color = 'chartreuse')
                plt.title(station_name)
                f.savefig(os.path.join(fn_fig, f'{station}_{station_name}_{dt}_ts_observed.png'), dpi=400)

        #%% MODELLED
        for station in ds.stations.values:
                print(station)
                #%%
                data_Tcal[station]['MOD'] = dict()
                station_name = dict_station_names[station]
                data_Tcal[station]['station_name'] = station_name
                print(f"Performing statistics for modelled ts for {station_name}")

                ts_peaks=pd.DataFrame()
                ts_dates = pd.DataFrame()
                i = 0
                for key in model_runs.keys():
                        print(i, key)
                        ts_key = ds.sel(stations = station, runs= key)['Q'].to_series() #runs_dict[key][station]
                        first_idx = ts_key.first_valid_index()
                        last_idx = ts_key.last_valid_index()
                        ts_key = ts_key.loc[first_idx:last_idx]

                        ts_peaks_value = ts_key.resample('AS-Oct').max()
                        ts_peaks_dates = ts_key.resample('AS-Oct').agg(lambda x : np.nan if x.count() == 0 else x.idxmax())
                        peaks = ts_key.loc[ts_peaks_dates.values[1:-1]]

                        #We perform the EVA - BM - Gumbel 
                        # peaks = pyex.get_extremes(ts_key, method = "BM", extremes_type="high", block_size="365.2426D", errors="raise")
                        
                        #Storing the date from the original AM time series 
                        dates = pd.DataFrame(columns=['month','date', 'member'])
                        dates['month'] = peaks.index.month
                        dates['date'] = peaks.index
                        dates['member'] = key

                        peaks.index  = peaks.index + pd.DateOffset(years=int(i)) 
                        ts_peaks = pd.concat([ts_peaks, peaks], axis = 0)
                        ts_dates = pd.concat([ts_dates, dates], axis = 0, ignore_index = True)
                        
                        print(len(ts_peaks))
                        i += len(ts_key.index.year.unique())
                ts_peaks.rename(columns={0:station}, inplace = True)
                if len(ts_peaks) == (64*16):
                        print("Date conversion seems ok!")
                #ts_peaks.sort_index(axis = 0, ascending=True, inplace = True)

                emp_T = calculate_emp(ts_peaks.values)
                emp_n1 = calculate_emp(ts_peaks.iloc[0:64].values) #We have 65 years
                emp_month = ts_dates['month']

                data_Tcal[station]['MOD']['peaks'] = ts_peaks
                data_Tcal[station]['MOD']['emp_T'] = emp_T
                data_Tcal[station]['MOD']['emp_month'] = emp_month
                data_Tcal[station]['MOD']['dates_model'] = ts_dates

                #Plotting the histogram
                #top_n = [10, 50, 104, 250, 500, 1040]
                top_n = [10, 50, 100, 250, 500, 1000]
                #Plotting the histogram of the highest values
                fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(8,10)) 
                for n, ax in zip(top_n, axs.reshape(-1)):
                        print(n)
                        ax.hist(ts_dates.loc[emp_T.sort_values(by='rank', ascending = True).iloc[0:n].index]['month'], bins=np.arange(14)-0.5, edgecolor = 'black', color='blue', stacked = True, density = True)
                        ax.set_xticks(range(13))
                        ax.set_title(f'Top {n} AM - {station_name}')
                        ax.set_xlim(0.5,12.5)
                fig.savefig(os.path.join(fn_fig, f'{station}_{station_name}_{dt}_top_hist_AM_events.png'), dpi=400)

               #Plotting the highest values per month
                fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(8,10)) 
                for n, ax in zip(top_n, axs.reshape(-1)):
                        print(n)

                        ax.plot(ts_dates.loc[emp_T.sort_values(by='rank', ascending = True).iloc[0:n].index]['month'],emp_T.sort_values(by='rank', ascending = True).iloc[0:n]['value'], marker = 'o', color='k', markersize=5, mew=0, lw=0, alpha=0.5)
                        #ax.hist(ts_dates.loc[emp_T.sort_values(by='rank', ascending = True).iloc[0:n].index]['month'], bins=np.arange(14)-0.5, edgecolor = 'black', color='blue', stacked = True, density = True), 
                        ax.set_xticks(range(13))
                        ax.set_title(f'Top {n} AM - {station_name}')
                        ax.set_xlim(0.5,12.5)
                        ax.set_ylabel('Discharge ($m^3/s$)', labelpad=-2)
                fig.savefig(os.path.join(fn_fig, f'{station}_{station_name}_{dt}_top_hist_value_AM_events.png'), dpi=400)

                #Plotting the highest values per month for n=1000
                fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(8,10), sharex=True, gridspec_kw={'height_ratios':[1,1.8],'hspace':0.05} ) 
                ax = axs.reshape(-1)
                ax[0].hist(ts_dates.loc[emp_T.sort_values(by='rank', ascending = True).iloc[0:n].index]['month'], bins=np.arange(14)-0.5, edgecolor = 'black', color='blue', stacked = True, density = True)
                ax[0].set_xticks(range(13))
                ax[0].set_title(f'Top 1000 AM - {station_name}')
                ax[0].set_ylabel('Relative frequency')
                # ax[0].set_xlim(0.5,12.5)
                ax[1].plot(ts_dates.loc[emp_T.sort_values(by='rank', ascending = True).iloc[0:1000].index]['month'],emp_T.sort_values(by='rank', ascending = True).iloc[0:1000]['value'], marker = 'o', color='k', markersize=5, mew=0, lw=0, alpha=0.5)
                #ax.hist(ts_dates.loc[emp_T.sort_values(by='rank', ascending = True).iloc[0:n].index]['month'], bins=np.arange(14)-0.5, edgecolor = 'black', color='blue', stacked = True, density = True), 
                ax[1].set_xticks(range(13))
                #ax.set_title(f'Top {n} AM - {station_name}')
                ax[1].set_xlim(0.5,12.5)
                ax[1].set_ylabel('Discharge ($m^3/s$)')
                ax[1].set_xlabel('Month')
                fig.savefig(os.path.join(fn_fig, f'{station}_{station_name}_{dt}_top_1000_hist_value_AM_events.png'), dpi=400)

                #%%
                #We sample some data length and calculate the statistics 
                bm_summary = dict()

                #Doing it per ensemble for genextreme and gumbel_r
                i_start = 0
                j = 1
                for n in np.arange(64, 1024+64, 64):
                        print(n)
                        i_end = n
                        print(i_end, i_start, j)

                        summ = pd.DataFrame(columns=['genextreme', 'gumbel_r', 'expon', 'genpareto'], index=['loc', 'scale', 'shape', 'AIC', 'rate']) 
                        ret_levels = pd.DataFrame(columns=['genextreme', 'gumbel_r', 'expon', 'genpareto'], index=Ts) 
                        cils = pd.DataFrame(columns=['genextreme', 'gumbel_r', 'expon', 'genpareto'], index=Ts) 
                        cihs = pd.DataFrame(columns=['genextreme', 'gumbel_r', 'expon', 'genpareto'], index=Ts) 

                        model = pyex.EVA.from_extremes(extremes = ts_peaks.iloc[i_start:i_end][station], method = "BM", extremes_type = 'high', block_size = "365.2425D")

                        summ[['gumbel_r']], ret_levels[['gumbel_r']], cils[['gumbel_r']], cihs[['gumbel_r']] = calc_stats(model, Ts, distr="gumbel_r")
                        summ[['genextreme']], ret_levels[['genextreme']], cils[['genextreme']], cihs[['genextreme']] = calc_stats(model, Ts, distr="genextreme")

                        bm_summary[f"65_r{j}"] = {'eva_params': summ, 'return_levels': ret_levels, 'conf_int_low': cils, 'conf_int_high': cihs}
                        i_start = i_end #i_end - 65 #i_start = 0!
                        j += 1
                
                #Doing it per concatenated ensemble for genextreme and gumbel_r
                for n in np.arange(64, 1024+64, 64):
                        print(n)
                        i_end = n
                        i_start = 0 #i_end - 65 #i_start = 0!
                        print(i_end, i_start)

                        summ = pd.DataFrame(columns=['genextreme', 'gumbel_r', 'expon', 'genpareto'], index=['loc', 'scale', 'shape', 'AIC', 'rate']) 
                        ret_levels = pd.DataFrame(columns=['genextreme', 'gumbel_r', 'expon', 'genpareto'], index=Ts) 
                        cils = pd.DataFrame(columns=['genextreme', 'gumbel_r', 'expon', 'genpareto'], index=Ts) 
                        cihs = pd.DataFrame(columns=['genextreme', 'gumbel_r', 'expon', 'genpareto'], index=Ts) 

                        model = pyex.EVA.from_extremes(extremes = ts_peaks.iloc[i_start:i_end][station], method = "BM", extremes_type = 'high', block_size = "365.2425D")

                        summ[['gumbel_r']], ret_levels[['gumbel_r']], cils[['gumbel_r']], cihs[['gumbel_r']] = calc_stats(model, Ts, distr="gumbel_r")
                        summ[['genextreme']], ret_levels[['genextreme']], cils[['genextreme']], cihs[['genextreme']] = calc_stats(model, Ts, distr="genextreme")

                        if (station == 9434) and (dt == 'daily'): #and (n == 1040):
                                summ[['genextreme']], ret_levels[['genextreme']], cils[['genextreme']], cihs[['genextreme']] = calc_stats_emcee(model, Ts, distr="genextreme")

                        bm_summary[str(n)] = {'eva_params': summ, 'return_levels': ret_levels, 'conf_int_low': cils, 'conf_int_high': cihs}

                data_Tcal[station]['MOD']['BM_fit'] = bm_summary
                #%%
                ######### PLOTTING #######################
                #For plotting
                max_y = roundup(max(bm_summary[str(1024)]['return_levels']['gumbel_r'].max(), bm_summary[str(1024)]['return_levels']['genextreme'].max(), emp_T.value.max(), bm_summary[str(1024)]['conf_int_high']['gumbel_r'].max(), bm_summary[str(1024)]['conf_int_high']['genextreme'].max()))

                distr='gumbel_r'
                f, ax = plt.subplots()
                plot_return_period_tradi(ax, cmap, norm, data_Tcal, ds_stats, [0, max_y], [1,2000], station, distr, station_name)
                plt.show()
                f.savefig(os.path.join(fn_fig, f'{station}_{station_name}_{dt}_{distr}_return_curve_tradi.png'), dpi=400)

                distr='genextreme'
                f, ax = plt.subplots()
                plot_return_period_tradi(ax, cmap, norm, data_Tcal, ds_stats, [0, max_y], [1,2000], station, distr, station_name)
                plt.show()
                f.savefig(os.path.join(fn_fig, f'{station}_{station_name}_{dt}_{distr}_return_curve_tradi.png'), dpi=400)

                #All years - 1040 years
                n=1024
                f, ax = plt.subplots()
                plot_gumbel_gev_n(ax, cmap, norm, data_Tcal, ds_stats, n, [0, max_y], [1,2000], station, f"{station_name} - {n} years")
                f.savefig(os.path.join(fn_fig, f'{station}_{station_name}_{dt}_gumbel_gev_return_curve_{n}.png'), dpi=400)

                #One ensemble - 65 years
                n=64
                f, ax = plt.subplots()
                plot_gumbel_gev_n(ax, cmap, norm, data_Tcal, ds_stats, n, [0, max_y], [1,2000], station, f"{station_name} - {n} years")
                f.savefig(os.path.join(fn_fig, f'{station}_{station_name}_{dt}_gumbel_gev_return_curve_{n}.png'), dpi=400)

                T_considered = [10,50,100,500,1000]
                cols = [ str(n) for n in np.arange(64, 1024+64, 64)]
                distr = 'gumbel_r'
                f, ax = plt.subplots()
                plot_convergence_params(ax, data_Tcal, station, T_considered, cols, distr, f"{station_name} - {distr}")
                f.savefig(os.path.join(fn_fig, f'{station}_{station_name}_{dt}_{distr}_return_level_conv.png'), dpi=400)

                distr = 'genextreme'
                f, ax = plt.subplots()
                plot_convergence_params(ax, data_Tcal, station, T_considered, cols, distr, f"{station_name} - {distr}")
                f.savefig(os.path.join(fn_fig, f'{station}_{station_name}_{dt}_{distr}_return_level_conv.png'), dpi=400)

                distr = 'genextreme'
                f, ax = plt.subplots()
                plot_convergence_params_type(ax, data_Tcal, station, cols, distr, 'shape', f"{station_name} - {distr}")
                f.savefig(os.path.join(fn_fig, f'{station}_{station_name}_{dt}_{distr}_shape_conv.png'), dpi=400)

                f, ax = plt.subplots()
                plot_convergence_params_type(ax, data_Tcal, station, cols, distr, 'scale', f"{station_name} - {distr}")
                f.savefig(os.path.join(fn_fig, f'{station}_{station_name}_{dt}_{distr}_scale_conv.png'), dpi=400)

                f, ax = plt.subplots()
                plot_convergence_params_type(ax, data_Tcal, station, cols, distr, 'loc', f"{station_name} - {distr}")
                f.savefig(os.path.join(fn_fig, f'{station}_{station_name}_{dt}_{distr}_location_conv.png'), dpi=400)

                plt.close('all')

        all_results[dt][source]['results'] = data_Tcal

#We save the dictionary
fn_out = os.path.join(fn_data_out, f'{dt}_results_stations.pickle')
with open(fn_out, 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(all_results, f, pickle.HIGHEST_PROTOCOL)








#%%
#  
#%% Compare outputs
# params_gumb = gumbel_r.fit(model.extremes.values)
# params_ic = genextreme.fit(model.extremes.values, loc=1324, scale=478)
# params_floc = genextreme.fit(model.extremes.values, floc=1324)

# #Generate p_exc from it
# p_cdf = np.linspace(0.001,0.999,1000)
# ts_ic = genextreme.isf(p_cdf, c=params_ic[0], loc=params_ic[1], scale=params_ic[2])
# ts_floc = genextreme.isf(p_cdf, c=params_floc[0], loc=params_floc[1], scale=params_floc[2]) # does not work!
# ts_gumb = gumbel_r.isf(p_csdf, loc=params_gumb[0], scale=params_gumb[1])

# f, ax = plt.subplots()
# ax.plot(1/(p_cdf), ts_ic, '.-r')
# ax.plot(1/(p_cdf), ts_floc, '.-b')
# ax.plot(1/(p_cdf), ts_gumb, '.-k')
# ax.set_xscale('log')
# ax.set_ylim(500,6000)
# plt.show()

#%% Highlighting the colors of the event
#%%
# #%% For now we select one and extract the extremes
# from pyextremes import (
#     __version__,
#     EVA,
#     plot_mean_residual_life,
#     plot_parameter_stability,
#     plot_return_value_stability,
#     plot_threshold_stability,
# )

# ts_pot=pd.DataFrame()
# i = 0
# for key in model_runs.keys():
#         print(i, key)
#         ts_key = runs_dict[key][station]
#         #Selecting threshold: https://github.com/georgebv/pyextremes/blob/898f132307e42316a6f1ef46a4264aff88d6754d/src/pyextremes/tuning/threshold_selection.py

        # #Parameter Stability:
        # ax = plot_mean_residual_life(ts_key)
        # fig = ax.get_figure()
        # plt.show()

#         #Parameter Stability:
#         fig = plt.figure()
#         ax = plot_parameter_stability(ts_key)
#         plt.show()

#         #Return value stability:
#         ax = plot_return_value_stability(ts_key)
#         plt.show()
#         fig = ax.get_figure()

#         #Threshold stability:
#         ax = plot_threshold_stability(ts_key)
#         plt.show()
#         fig = ax.get_figure()


#         #We perform the EVA - POT - GP 
#         peaks = pyex.get_extremes(ts_key, method = 'POT', extremes_type="high", block_size="365.2426D", errors="raise")
#         peaks.index  = peaks.index + pd.DateOffset(years=int(i)) 
#         ts_peaks = pd.concat([ts_peaks, peaks], axis = 0)
#         print(len(ts_peaks))
#         i += len(ts_key.index.year.unique())
# ts_pot.rename(columns={0:station}, inplace = True)
# ts_pot.sort_index(axis = 0, ascending=True, inplace = True)

#%%
# summary = model.get_summary(return_period=[1.1, 2, 5, 10, 25, 50, 100, 250, 500, 1000],
#                             alpha=0.95,
#                             n_samples=None)

# # %% Selecting peaks
# #We perform the EVA - BM - Gumbel 
# # peaks = pyex.get_extremes(ts, method = "BM", extremes_type="high", block_size="365.2426D", errors="raise")

# #%% Initializing model EVA from extremes
# model = pyex.EVA.from_extremes(extremes = peaks.reset_index(drop=True), method = "BM", extremes_type = 'high', block_size = "365.2425D")
# #model = pyex.EVA(data = ts)

# model.get_extremes(method = "BM", extremes_type="high", block_size="365.2426D", errors="raise")
# #model.plot_extremes()
# #ts_bm = model.extremes

# # Fitting EVA model
# model.fit_model(model = "MLE", distribution = ['genextreme', 'gumbel_r']) #Emcee

# #%%We perform the EVA - BM - GEV
# #model = pyex.EVA(data = ts)
# model.get_extremes(method = "BM", extremes_type="high", block_size="365.2426D", errors="raise")
# #model.plot_extremes()
# #ts_bm = model.extremes

# # Fitting EVA model
# model.fit_model(model = "MLE", distribution = 'genextreme') #Emcee

# #%% We look at POT


# #%%
# summary = model.get_summary(
#     return_period=[1.1, 2, 5, 10, 25, 50, 100, 250, 500, 1000],
#     alpha=0.95,
#     n_samples=100,
# )
#model.plot_diagnostic(alpha=0.95) #Plots the return values with other confidence plots






#%%Checking results with other sampling
# smax = ts.resample("AS-SEP").max() # hydrological max
# ymax = ts.resample("AS").max() # yearly max

#Plotting results
# model.get_extremes()
# pyex.plotting.plot_extremes(ts, extremes = ts_bm, extremes_method = "BM", extremes_type = "high", block_size = "365.2426D")


#%%Calculating empirical return periods
# t_emp = pyex.get_return_periods(
#     ts=ts,
#     extremes=model.extremes,
#     extremes_method="BM",
#     extremes_type="high",
#     block_size="365.2425D",
#     return_period_size="365.2425D",
#     plotting_position="weibull",
# )
#t_emp.sort_values("return period", ascending=False).head()
# %%




#Official statistics
# obs_t = {'Q_1011': retlev_hydro.isel(station=1).to_dataframe()[['valeur','int_conf_bas', 'int_conf_haut']], #hydrofrance
#         'Q_4': retlev_hydro.isel(station=0).to_dataframe()[['valeur','int_conf_bas', 'int_conf_haut']], #hydrofrance et SPW
#         'Q_16': 0, #/St Pieter -- station recalculée Discharge at Kanne + discharge at St Pieter 
#         'Q_12': retlev_spw.isel(station=5).to_dataframe()[['valeur','int_conf_bas', 'int_conf_haut']],# SPW - severely hit 2021
#         'Q_11': retlev_spw.isel(station=0).to_dataframe()[['valeur','int_conf_bas', 'int_conf_haut']],# SPW 
#         'Q_10':  retlev_spw.isel(station=2).to_dataframe()[['valeur','int_conf_bas', 'int_conf_haut']],# SPW 
#         'Q_801':  retlev_spw.isel(station=1).to_dataframe()[['valeur','int_conf_bas', 'int_conf_haut']], # SPW
#         "Q_9": 0, #SPW
#         'Q_101':0,
#         # Roer
#         # Geul 
#         } 

# #Original data - hourly or daily data
# if dt == 'daily':
#         data_t = {'Q_1011': hydro_daily['Q'].isel(wflow_id=0).to_series(), #hydrofrance - Meuse at Goncourt
#                 'Q_4': hydro_daily['Q'].isel(wflow_id=2).to_series(), #hydrofrance et SPW - "Meuse at Chooz"
#                 'Q_16': 0, #"Meuse at Borgharen"
#                 'Q_12': 0,# SPW - severely hit 2021 - "Vesdre at Chaudfontaine"
#                 'Q_11': 0,# SPW - "Ambleve at Martinrive"
#                 'Q_10':  0,# SPW - "Ourthe at Tabreux"
#                 'Q_801':  0, # SPW - "Lesse at Gendron"
#                 "Q_9": 0, #SPW - "Sambre at Salzinnes", 
#                 "Q_101": 0, # "Meuse at Saint Mihiel"
#                 # Roer
#                 # Geul 
#                 } 
# if dt == 'hourly':
#         data_t = {'Q_1011': 0, #hydrofrance - Meuse at Goncourt
#                 'Q_4': 0, #hydrofrance et SPW - "Meuse at Chooz"
#                 'Q_16': 0, #"Meuse at Borgharen"
#                 'Q_12': spw_hourly['Q'].sel(id=6228).to_series(),# SPW - severely hit 2021 - "Vesdre at Chaudfontaine"
#                 'Q_11': spw_hourly['Q'].sel(id=6621).to_series(),# SPW - "Ambleve at Martinrive"
#                 'Q_10':  spw_hourly['Q'].sel(id=5921).to_series(),# SPW - "Ourthe at Tabreux"
#                 'Q_801':  spw_hourly['Q'].sel(id=8221).to_series(), # SPW - "Lesse at Gendron"
#                 "Q_9": spw_hourly['Q'].sel(id=7319).to_series(), #SPW - "Sambre at Salzinnes", 
#                 "Q_101": 0, # "Meuse at Saint Mihiel"
#                 # Roer
#                 # Geul 
#                 } 

#%%Important locations
# locs = {'Q_1011': "Meuse at Goncourt", #hydrofrance
#         'Q_4': "Meuse at Chooz", #hydrofrance et SPW
#         'Q_16': "Meuse at Borgharen", #/St Pieter -- station recalculée Discharge at Kanne + discharge at St Pieter 
#         'Q_12': "Vesdre at Chaudfontaine",# SPW - severely hit 2021
#         'Q_11': "Ambleve at Martinrive",# SPW 
#         'Q_10': "Ourthe at Tabreux",# SPW 
#         'Q_801': "Lesse at Gendron", # SPW
#         "Q_9": "Sambre at Salzinnes", #SPW
#         "Q_101": "Meuse at St-Mihiel" 

#         # Roer
#         # Geul 
#         } 

# use_cols = ['time'] + list(locs.keys()) 

# shp_catch = {'Q_1011': "subcatch_S01", #hydrofrance
#         'Q_4': "subcatch_S04", #hydrofrance et SPW
#         'Q_16': 'subcatch_S06', #/St Pieter -- station recalculée Discharge at Kanne + discharge at St Pieter 
#         'Q_12': "subcatch_S01",# SPW - severely hit 2021
#         'Q_11': "subcatch_S01",# SPW 
#         'Q_10': "subcatch_S02",# SPW 
#         'Q_801': "subcatch_S02", # SPW
#         "Q_9": "subcatch_S02", #SPW
#         "Q_101": "subcatch_S02",
#         # Roer
#         # Geul 
#         } 
#%% We read the observed data

#For the official statistics sheet
# locs_spw = {6228 : "Vesdre at Chaudfontaine",# SPW - old Q_12 - severely hit 2021
#         6621 : "Ambleve at Martinrive",# SPW - old Q_11
#         5921 : "Ourthe at Tabreux",# SPW - old Q_10
#         8221 : "Lesse at Gendron", # SPW - old Q_801
# #        "": "Sambre at Salzinnes", #SPW - old Q_9 - cannot find official statistics
#         9434 : "Semois at Membre"}
# #        "" : "Sambre at Floriffoux"} - cannot find ID in the nc file

#%% We read the observed data

# #SPW - official statistics
# fn_spw = '/p/11208719-interreg/data/spw/statistiques/c_final/spw_statistics.nc' #spw_statistiques doesn't exist anymore
# retlev_spw = xr.open_dataset(fn_spw)

# #SPW - raw HOURLY data
# fn_spw_raw = '/p/11208719-interreg/data/spw/Discharge/c_final/hourly_spw_discharges.nc' 
# spw_hourly = xr.open_dataset(fn_spw_raw).load()

# #For the official statistics sheet
# locs_spw = {6228 : "Vesdre at Chaudfontaine",# SPW - old Q_12 - severely hit 2021
#         6621 : "Ambleve at Martinrive",# SPW - old Q_11
#         5921 : "Ourthe at Tabreux",# SPW - old Q_10
#         8221 : "Lesse at Gendron", # SPW - old Q_801
# #        "": "Sambre at Salzinnes", #SPW - old Q_9 - cannot find official statistics
#         9434 : "Semois at Membre"}
# #        "" : "Sambre at Floriffoux"} - cannot find ID in the nc file

# #HYDRO
# fn_hydro = '/p/11208719-interreg/data/hydroportail/c_final/hydro_statistiques_daily.nc'
# retlev_hydro = xr.open_dataset(fn_hydro)
# hydro_daily = xr.open_dataset('/p/11208719-interreg/data/hydroportail/c_final/hydro_daily.nc').load()

#%% We perform the statistics on the observations
# data_Tcal = dict()
# for station in locs.keys():
#         print(station)
#         #
#         #station = 'Q_9' 
#         #
#         print(locs[station], station)
#         if isinstance(data_t[station], pd.Series):
#                 print(f"Performing statistics on observed ts for {station}")
#                 #For now we select one and extract the extremes
#                 ts_peaks=pd.DataFrame()
#                 ts_key = data_t[station] #We need to remove years with full nans!!
#                 first_idx = ts_key.first_valid_index()
#                 last_idx = ts_key.last_valid_index()
#                 ts_key = ts_key.loc[first_idx:last_idx]

#                 #We perform the EVA - BM - Gumbel 
#                 peaks = pyex.get_extremes(ts_key, method = "BM", extremes_type="high", block_size="365.2426D", errors="ignore")
#                 ts_peaks = peaks.copy()

#                 emp_T = calculate_emp(ts_peaks.values)
#                 emp_month = ts_peaks.index.month

#                 data_Tcal[station] = dict()
#                 data_Tcal[station]['peaks'] = ts_peaks
#                 data_Tcal[station]['emp_T'] = emp_T
#                 data_Tcal[station]['emp_month'] = emp_month

#                 Ts = [1.1, 1.5, 1.8, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 1200]
#                 summ = pd.DataFrame(columns=['genextreme', 'gumbel_r', 'expon', 'genpareto'], index=['loc', 'scale', 'shape', 'AIC', 'rate']) 
#                 ret_levels = pd.DataFrame(columns=['genextreme', 'gumbel_r', 'expon', 'genpareto'], index=Ts) 
#                 cils = pd.DataFrame(columns=['genextreme', 'gumbel_r', 'expon', 'genpareto'], index=Ts) 
#                 cihs = pd.DataFrame(columns=['genextreme', 'gumbel_r', 'expon', 'genpareto'], index=Ts) 

#                 model = pyex.EVA.from_extremes(extremes = ts_peaks, method = "BM", extremes_type = 'high', block_size = "365.2425D")

#                 summ[['gumbel_r']], ret_levels[['gumbel_r']], cils[['gumbel_r']], cihs[['gumbel_r']] = calc_stats(model, Ts, distr="gumbel_r")
#                 summ[['genextreme']], ret_levels[['genextreme']], cils[['genextreme']], cihs[['genextreme']] = calc_stats(model, Ts, distr="genextreme")

#                 data_Tcal[station]['BM_fit'] = {'eva_params': summ, 'return_levels': ret_levels, 'conf_int_low': cils, 'conf_int_high': cihs}

#                 #We plot the time series
#                 f, ax = plt.subplots()
#                 data_t[station].dropna().plot(ax=ax, lw=0.5)
#                 ax.set_ylabel('Discharge ($m^3/s$)')
#                 ax.set_ylim(bottom=0)
#                 if isinstance(max_2021[station], float): #Observed time series
#                         ax.hlines(y=max_2021[station], xmin=data_t[station].dropna().index[0], xmax=data_t[station].dropna().index[-1], color = 'chartreuse')
#                 plt.title(locs[station])
#                 f.savefig(os.path.join(fn_fig, f'{station}_{dt}_ts_observed.png'), dpi=400)

#%%
                # #Plotting results - return levels per 65 years ensemble
                # f, ax = plt.subplots()
                # distr = 'gumbel_r'
                # for n in np.arange(1, 17, 1):
                #         print(n)
                #         ax.plot(bm_summary[f"65_r{n}"]['return_levels'].index, bm_summary[f"65_r{n}"]['return_levels'][distr], '-k', zorder = 2)
                #         ax.fill_between(bm_summary[f"65_r{n}"]['return_levels'].index.values, bm_summary[f"65_r{n}"]['conf_int_low'][distr].values.astype(float), bm_summary[f"65_r{n}"]['conf_int_high'][distr].values.astype(float), color='grey', edgecolor=None, alpha=0.1, zorder = 1)
                
                # if isinstance(ds_stats, xr.Dataset): #Official statistics
                #         df_stats = ds_stats.sel(stations=station).to_dataframe()
                #         ax.plot(df_stats.index, df_stats['valeur'].values, 'or', zorder = 5, label = 'official stat.')
                #         ax.vlines(x= df_stats.index, ymin = df_stats['int_conf_bas'].values, ymax = df_stats['int_conf_haut'].values, color='r', zorder = 5)                
                # #Observed time series
                # ax.scatter(data_Tcal[station]['emp_T'].emp_rp.values, data_Tcal[station]['emp_T'].value, c = data_Tcal[station]['emp_month'].values, marker = 'x', label='observed', cmap = cmap, norm=norm, zorder = 7)
                
                # # if isinstance(max_2021[station], float): #Observed 2021 level
                # #         ax.hlines(y=max_2021[station], xmin=min(Ts), xmax=max(Ts), color = 'chartreuse', label = '2021 level')
                # m = ax.scatter(emp_T.emp_rp.values, emp_T.value, c = emp_month.values, marker = '.', label='modelled', cmap = cmap, norm=norm, zorder = 6)        
                # ax.plot(bm_summary[str(1040)]['return_levels'].index, bm_summary[str(1040)]['return_levels'][distr], '-r', label='Gumbel', zorder = 4)
                # ax.fill_between(bm_summary[str(1040)]['return_levels'].index.values, bm_summary[str(1040)]['conf_int_low'][distr].values.astype(float), bm_summary[str(1040)]['conf_int_high'][distr].values.astype(float), color='darkred', edgecolor=None, alpha=0.5, zorder = 3)        
                # ax.set_xscale('log')
                # ax.set_ylim(0, max_y)
                # ax.set_xlim(1.0,1200)
                # ax.set_xlabel('Return Period (year)')
                # ax.set_ylabel('Discharge ($m^3/s$)')
                # ax.legend()
                # plt.title(station_name)
                # plt.show()
                # f.savefig(os.path.join(fn_fig, f'{station}_{dt}_{distr}_return_curve_tradi.png'), dpi=400)

                # f, ax = plt.subplots()
                # distr = 'genextreme'
                # for n in np.arange(1, 17, 1):
                #         print(n)
                #         ax.plot(bm_summary[f"65_r{n}"]['return_levels'].index, bm_summary[f"65_r{n}"]['return_levels'][distr], '-k', zorder = 2)
                #         ax.fill_between(bm_summary[f"65_r{n}"]['return_levels'].index.values, bm_summary[f"65_r{n}"]['conf_int_low'][distr].values.astype(float), bm_summary[f"65_r{n}"]['conf_int_high'][distr].values.astype(float), color='grey', edgecolor=None, alpha=0.1, zorder = 1)
                # if isinstance(obs_t[station], pd.DataFrame): #Official statistics
                #         ax.plot(obs_t[station].index, obs_t[station]['valeur'].values, 'or', zorder = 5, label = 'official stat.')
                #         ax.vlines(x= obs_t[station].index, ymin = obs_t[station]['int_conf_bas'].values, ymax = obs_t[station]['int_conf_haut'].values, color='r', zorder = 5)
                # if isinstance(data_t[station], pd.Series): #Observed time series
                #         ax.scatter(data_Tcal[station]['emp_T'].emp_rp.values, data_Tcal[station]['emp_T'].value, c = data_Tcal[station]['emp_month'].values, marker = 'x', label='observed', cmap = cmap, norm=norm, zorder = 7)
                # if isinstance(max_2021[station], float): #Observed time series
                #         ax.hlines(y=max_2021[station], xmin=min(Ts), xmax=max(Ts), color = 'chartreuse', label = '2021 level')        
                # m = ax.scatter(emp_T.emp_rp.values, emp_T.value, c = emp_month.values, marker = '.', label='modelled', cmap = cmap, norm=norm, zorder = 6)        
                # ax.plot(bm_summary[str(1040)]['return_levels'].index, bm_summary[str(1040)]['return_levels'][distr], '-r', label='GEV', zorder = 4)
                # ax.fill_between(bm_summary[str(1040)]['return_levels'].index.values, bm_summary[str(1040)]['conf_int_low'][distr].values.astype(float), bm_summary[str(1040)]['conf_int_high'][distr].values.astype(float), color='darkred', edgecolor=None, alpha=0.5, zorder = 3)        
                # ax.set_xscale('log')
                # ax.set_ylim(0, max_y)
                # ax.set_xlim(1.0,1200)
                # ax.set_xlabel('Return Period (year)')
                # ax.set_ylabel('Discharge ($m^3/s$)')
                # ax.legend()
                # plt.title(locs[station])
                # plt.show()
                # f.savefig(os.path.join(fn_fig, f'{station}_{dt}_{distr}_return_curve_tradi.png'), dpi=400)

                # #All years - 1040 years
                # n=1040
                # f, ax = plt.subplots()
                # ax.plot(bm_summary[str(n)]['return_levels'].index, bm_summary[str(n)]['return_levels']['gumbel_r'], '-b', label='Gumbel')
                # ax.plot(bm_summary[str(n)]['return_levels'].index, bm_summary[str(n)]['return_levels']['genextreme'], '-g', label='GEV')
                # ax.fill_between(bm_summary[str(n)]['return_levels'].index.values, bm_summary[str(n)]['conf_int_low']['gumbel_r'].values.astype(float), bm_summary[str(n)]['conf_int_high']['gumbel_r'].values.astype(float), color='blue', edgecolor=None, alpha=0.1)
                # ax.fill_between(bm_summary[str(n)]['return_levels'].index.values, bm_summary[str(n)]['conf_int_low']['genextreme'].values.astype(float), bm_summary[str(n)]['conf_int_high']['genextreme'].values.astype(float), color='green', edgecolor=None, alpha=0.1)
                # m = ax.scatter(emp_T.emp_rp.values, emp_T.value, c = emp_month.values, marker = '.', label='modelled', cmap = cmap, norm=norm)
                # if isinstance(data_t[station], pd.Series):
                #         ax.scatter(data_Tcal[station]['emp_T'].emp_rp.values, data_Tcal[station]['emp_T'].value, c = data_Tcal[station]['emp_month'].values, marker = 'x', label='observed', cmap = cmap, norm=norm)
                # if isinstance(obs_t[station], pd.DataFrame):
                #         ax.plot(obs_t[station].index, obs_t[station]['valeur'].values, 'or', label='official statistics')
                #         ax.vlines(x= obs_t[station].index, ymin = obs_t[station]['int_conf_bas'].values, ymax = obs_t[station]['int_conf_haut'].values, color='r')
                # if isinstance(max_2021[station], float): #Observed time series
                #         ax.hlines(y=max_2021[station], xmin=min(Ts), xmax=max(Ts), color = 'chartreuse', label = '2021 level')        
                # ax.set_xscale('log')
                # ax.set_ylim(0, max_y)
                # ax.set_xlim(1.0,1200)
                # ax.set_xlabel('Return Period (year)')
                # ax.set_ylabel('Discharge (m3/s)')
                # # produce a legend with the unique colors from the scatter
                # legend1 = ax.legend(*m.legend_elements(), title="Month", loc="upper left", fontsize = 7)
                # #ax.add_artist(legend1)
                # ax.legend()
                # plt.title(locs[station] + f' - {n} years')
                # plt.show()
                # f.savefig(os.path.join(fn_fig, f'{station}_{dt}_gumbel_gev_return_curve_{n}.png'), dpi=400)

                # #Both - 65 years
                # n=65
                # f, ax = plt.subplots()
                # ax.plot(bm_summary[str(n)]['return_levels'].index, bm_summary[str(n)]['return_levels']['gumbel_r'], '-b', label='Gumbel')
                # ax.plot(bm_summary[str(n)]['return_levels'].index, bm_summary[str(n)]['return_levels']['genextreme'], '-g', label='GEV')
                # ax.fill_between(bm_summary[str(n)]['return_levels'].index.values, bm_summary[str(n)]['conf_int_low']['gumbel_r'].values.astype(float), bm_summary[str(n)]['conf_int_high']['gumbel_r'].values.astype(float), color='blue', edgecolor=None, alpha=0.1)
                # ax.fill_between(bm_summary[str(n)]['return_levels'].index.values, bm_summary[str(n)]['conf_int_low']['genextreme'].values.astype(float), bm_summary[str(n)]['conf_int_high']['genextreme'].values.astype(float), color='green', edgecolor=None, alpha=0.1)
                # m = ax.scatter(emp_n1.emp_rp.values, emp_n1.value, c = emp_month[0:65].values, marker = '.', label='modelled', cmap = cmap, norm=norm)
                # if isinstance(data_t[station], pd.Series):
                #         ax.scatter(data_Tcal[station]['emp_T'].emp_rp.values, data_Tcal[station]['emp_T'].value, c = data_Tcal[station]['emp_month'].values, marker = 'x', label='observed', cmap = cmap, norm=norm)
                # if isinstance(obs_t[station], pd.DataFrame):
                #         ax.plot(obs_t[station].index, obs_t[station]['valeur'].values, 'or', label='official statistics')
                #         ax.vlines(x= obs_t[station].index, ymin = obs_t[station]['int_conf_bas'].values, ymax = obs_t[station]['int_conf_haut'].values, color='r')
                # if isinstance(max_2021[station], float): #Observed time series
                #         ax.hlines(y=max_2021[station], xmin=min(Ts), xmax=max(Ts), color = 'chartreuse', label = '2021 level')        
                # ax.set_xscale('log')
                # ax.set_ylim(0, max_y)
                # ax.set_xlim(1.0,1200)
                # ax.set_xlabel('Return Period (year)')
                # ax.set_ylabel('Discharge (m3/s)')
                # legend1 = ax.legend(*m.legend_elements(), title="Month", loc="upper left", fontsize = 7)
                # #ax.add_artist(legend1)
                # ax.legend()
                # plt.title(locs[station] + f' - {n} years')
                # plt.show()
                # f.savefig(os.path.join(fn_fig, f'{station}_{dt}_gumbel_gev_return_curve_{n}.png'), dpi=400)

        #        #
        #         T_considered = [10,50,100,500,1000]
        #         cols = [ str(n) for n in np.arange(65, 1040+65, 65)]


        #         #Collecting the other way
        #         distr = 'genextreme'
        #         color=iter(cm.rainbow(np.linspace(0,1,len(T_considered)+1)))
        #         f, ax = plt.subplots()
        #         for T_i in T_considered:
        #                 c=next(color)
        #                 unc = pd.DataFrame(columns = cols, index= ['ret_value', 'cil', 'cih'])
        #                 for key in cols:
        #                         unc.loc['ret_value',key]=  bm_summary[key]['return_levels'].loc[T_i, distr]
        #                         unc.loc['cil',key]=  bm_summary[key]['conf_int_low'].loc[T_i, distr]
        #                         unc.loc['cih',key]=  bm_summary[key]['conf_int_high'].loc[T_i, distr]
        #                 ax.plot([int(i) for i in unc.columns],unc.loc['ret_value',:],'o-', color=c, label = str(T_i))
        #                 ax.fill_between([int(i) for i in unc.columns],unc.loc['cil',:].values.astype(float),unc.loc['cih',:].values.astype(float),color=c, alpha=0.15, edgecolor=None)
        #                 if isinstance(obs_t[station], pd.DataFrame):
        #                         try:
        #                                 ax.plot([int(i) for i in unc.columns], np.repeat(obs_t[station]['valeur'].loc[float(T_i)], len([int(i) for i in unc.columns])), '--', color=c)
        #                         except:
        #                                 continue
        #         ax.legend()
        #         ax.set_xlabel("Record length (years)")
        #         ax.set_ylabel("Discharge (m3/s)")
        #         plt.title(locs[station]+' - '+ distr)
        #         #ax.set_ylim(2000, 6000)
        #         plt.show()
        #         f.savefig(os.path.join(fn_fig, f'{station}_{dt}_{distr}_return_level_conv.png'), dpi=400)

        #         #Collecting the other way
        #         distr = 'gumbel_r'
        #         color=iter(cm.rainbow(np.linspace(0,1,len(T_considered)+1)))
        #         f, ax = plt.subplots()
        #         for T_i in T_considered:
        #                 c=next(color)
        #                 unc = pd.DataFrame(columns = cols, index= ['ret_value', 'cil', 'cih'])
        #                 for key in cols:
        #                         unc.loc['ret_value',key]=  bm_summary[key]['return_levels'].loc[T_i, distr]
        #                         unc.loc['cil',key]=  bm_summary[key]['conf_int_low'].loc[T_i, distr]
        #                         unc.loc['cih',key]=  bm_summary[key]['conf_int_high'].loc[T_i, distr]
        #                 ax.plot([int(i) for i in unc.columns],unc.loc['ret_value',:],'o-', color=c, label = str(T_i))
        #                 ax.fill_between([int(i) for i in unc.columns],unc.loc['cil',:].values.astype(float),unc.loc['cih',:].values.astype(float),color=c, alpha=0.15, edgecolor=None)
        #                 # ax.plot([int(i) for i in unc.columns],unc.loc['cil',:],'--k')
        #                 # ax.plot([int(i) for i in unc.columns],unc.loc['cih',:], '--k')
        #                 if isinstance(obs_t[station], pd.DataFrame):
        #                         try:
        #                                 ax.plot([int(i) for i in unc.columns], np.repeat(obs_t[station]['valeur'].loc[float(T_i)], len([int(i) for i in unc.columns])), '--', color=c)
        #                         except:
        #                                 continue
        #         ax.legend()
        #         ax.set_xlabel("Record length (years)")
        #         ax.set_ylabel("Discharge (m3/s)")
        #         plt.title(locs[station]+' - '+distr)
        #         #ax.set_ylim(2000, 6000)
        #         plt.show()
        #         f.savefig(os.path.join(fn_fig, f'{station}_{dt}_{distr}_return_level_conv.png'), dpi=400)
