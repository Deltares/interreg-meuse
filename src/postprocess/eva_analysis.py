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
import math
from matplotlib.pyplot import cm
# %% Defining functions
def roundup(x):
        return int(math.ceil(x / 100.0)) * 100

def calc_stats(model, Ts, distr="gumbel_r"):
        summ = pd.DataFrame(columns=[distr], index=['loc', 'scale', 'shape', 'AIC', 'rate']) 
        ret_levels = pd.DataFrame(columns=[distr], index=Ts) 
        cils = pd.DataFrame(columns=[distr], index=Ts) 
        cihs = pd.DataFrame(columns=[distr], index=Ts) 

        if distr == 'genextreme': #We correct the starting conditions from gumbel_r
                params_gumb = gumbel_r.fit(model.extremes.values)
                params_gev = genextreme.fit(model.extremes.values, loc = params_gumb[0], scale=params_gumb[1])
                model.fit_model(model = "MLE", distribution = distr, distribution_kwargs={'floc': params_gev[1], 'fscale': params_gev[2]}) 
                summ.loc['loc', distr] = model.distribution.fixed_parameters['floc']
                summ.loc['scale', distr] = model.distribution.fixed_parameters['fscale']
        else:
                model.fit_model(model = "MLE", distribution = distr) #try also Emcee at some point
                summ.loc['loc', distr] = model.distribution.mle_parameters['loc']
                summ.loc['scale', distr] = model.distribution.mle_parameters['scale']
        summ.loc['AIC', distr] = model.AIC
        if 'c' in model.distribution.mle_parameters.keys():
                summ.loc['shape', distr] = model.distribution.mle_parameters['c']
        summ.loc['rate', distr] = 1.0
        ret_lev, cil, cih = model.get_return_value(Ts, return_period_size = "365.2425D", alpha=0.95, n_samples=100)
        ret_levels.loc[:,distr]=ret_lev
        cils.loc[:,distr]=cil
        cihs.loc[:,distr]=cih
        return summ, ret_levels, cils, cihs 
#%%
# We import the modelled data
Folder_start = "/p/11208719-interreg"
model_wflow = "f_spwgauges"
Folder_p = os.path.join(Folder_start, "wflow", model_wflow)
folder = "members_bias_corrected_daily"
fn_fig = os.path.join(Folder_start, "Figures", model_wflow, folder)

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
#%%Important locations
locs = {'Q_1011': "Meuse at Goncourt", #hydrofrance
        'Q_4': "Meuse at Chooz", #hydrofrance et SPW
        'Q_16': "Meuse at Borgharen", #/St Pieter -- station recalculée Discharge at Kanne + discharge at St Pieter 
        'Q_12': "Vesdre at Chaudfontaine",# SPW - severely hit 2021
        'Q_11': "Ambleve at Martinrive",# SPW 
        'Q_10': "Ourthe at Tabreux",# SPW 
        'Q_801': "Lesse at Gendron", # SPW
        "Q_9": "Sambre at Salzinnes", #SPW
        # Roer
        # Geul 
        } 

use_cols = ['time'] + list(locs.keys()) 
#%% Other parameters for POT
thr = {'Q_1011': 35, #hydrofrance
        'Q_4': 800, #hydrofrance et SPW
        'Q_16': 1000, #/St Pieter -- station recalculée Discharge at Kanne + discharge at St Pieter 
        'Q_12': 80,# SPW - severely hit 2021
        'Q_11': 130,# SPW 
        'Q_10': 150,# SPW 
        'Q_801': 160, # SPW
        "Q_9": 250, #SPW
        # Roer
        # Geul 
        } 

r_decl = {'Q_1011': '7D', #hydrofrance
        'Q_4': '7D', #hydrofrance et SPW
        'Q_16': '7D', #/St Pieter -- station recalculée Discharge at Kanne + discharge at St Pieter 
        'Q_12': '7D',# SPW - severely hit 2021
        'Q_11': '7D',# SPW 
        'Q_10': '7D',# SPW 
        'Q_801': '7D', # SPW
        "Q_9": '7D', #SPW
        # Roer
        # Geul 
        } 

shp_catch = {'Q_1011': "subcatch_S01", #hydrofrance
        'Q_4': "subcatch_S04", #hydrofrance et SPW
        'Q_16': 'subcatch_S06', #/St Pieter -- station recalculée Discharge at Kanne + discharge at St Pieter 
        'Q_12': "subcatch_S01",# SPW - severely hit 2021
        'Q_11': "subcatch_S01",# SPW 
        'Q_10': "subcatch_S02",# SPW 
        'Q_801': "subcatch_S02", # SPW
        "Q_9": "subcatch_S02", #SPW
        # Roer
        # Geul 
        } 
#%% We read the observed data
fn_spw = '/p/11208719-interreg/data/spw/statistiques/c_final/spw_statistiques.nc'
retlev_spw = xr.open_dataset(fn_spw)

fn_hydro = '/p/11208719-interreg/data/hydroportail/c_final/hydro_statistiques_daily.nc'
retlev_hydro = xr.open_dataset(fn_hydro)

obs_t = {'Q_1011': retlev_hydro.isel(station=1)['valeur'].to_series(), #hydrofrance
        'Q_4': retlev_hydro.isel(station=0)['valeur'].to_series(), #hydrofrance et SPW
        'Q_16': 0, #/St Pieter -- station recalculée Discharge at Kanne + discharge at St Pieter 
        'Q_12': retlev_spw.isel(station=5)['valeur'].to_series(),# SPW - severely hit 2021
        'Q_11': retlev_spw.isel(station=0)['valeur'].to_series(),# SPW 
        'Q_10':  retlev_spw.isel(station=2)['valeur'].to_series(),# SPW 
        'Q_801':  retlev_spw.isel(station=1)['valeur'].to_series(), # SPW
        "Q_9": 0, #SPW
        # Roer
        # Geul 
        } 
#%%
# Storing all the results in one location 
runs_dict = {}
for key in model_runs.keys():
    print(key)
    case = model_runs[key]["case"]
    ens = model_runs[key]["folder"] 
    runs_dict[key] = pd.read_csv(os.path.join(Folder_p, case, ens, "output.csv"), index_col=['time'], header=0, usecols = use_cols, parse_dates=['time'], date_parser = date_parser)
#%%
station = 'Q_1011'
#%%
for station in locs.keys():
        print(station)
        #For now we select one and extract the extremes
        ts_peaks=pd.DataFrame()
        i = 0
        for key in model_runs.keys():
                print(i, key)
                ts_key = runs_dict[key][station]
                #We perform the EVA - BM - Gumbel 
                peaks = pyex.get_extremes(ts_key, method = "BM", extremes_type="high", block_size="365.2426D", errors="raise")
                peaks.index  = peaks.index + pd.DateOffset(years=int(i)) 
                ts_peaks = pd.concat([ts_peaks, peaks], axis = 0)
                print(len(ts_peaks))
                i += len(ts_key.index.year.unique())
        ts_peaks.rename(columns={0:station}, inplace = True)
        if len(ts_peaks) == 1040:
                print("Date conversion seems ok!")
        ts_peaks.sort_index(axis = 0, ascending=True, inplace = True)

        #We sample some data length and calculate the statistics 
        Ts = [1.1, 1.5, 1.8, 2, 5, 10, 25, 50, 100, 250, 500, 1000]
        bm_summary = dict()

        #Doing it per ensemble for genextreme and gumbel_r
        for n in np.arange(65, 1040+65, 65):
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

                bm_summary[str(n)] = {'eva_params': summ, 'return_levels': ret_levels, 'conf_int_low': cils, 'conf_int_high': cihs}

        #Plotting results - return levels per 65 years
        f, ax = plt.subplots()
        distr = 'gumbel_r'
        for n in np.arange(65, 65*1+65, 65):
                print(n)
                ax.plot(bm_summary[str(n)]['return_levels'].index, bm_summary[str(n)]['return_levels'][distr], '-k')
                ax.fill_between(bm_summary[str(n)]['return_levels'].index.values, bm_summary[str(n)]['conf_int_low'][distr].values.astype(float), bm_summary[str(n)]['conf_int_high'][distr].values.astype(float), color='grey', edgecolor=None, alpha=0.1)
        if isinstance(obs_t[station], pd.Series):
                ax.plot(obs_t[station].index, obs_t[station].values, 'or')
        ax.set_xscale('log')
        ax.set_ylim(0, roundup(bm_summary[str(n)]['return_levels'][distr].max()))
        ax.set_xlabel('Return Period (year)')
        ax.set_ylabel('Discharge (m3/s)')
        plt.title(locs[station])
        plt.show()
        f.savefig(os.path.join(fn_fig, f'{station}_{distr}_return_curve_tradi.png'), dpi=400)

        f, ax = plt.subplots()
        distr = 'genextreme'
        for n in np.arange(65, 65*1+65, 65):
                print(n)
                ax.plot(bm_summary[str(n)]['return_levels'].index, bm_summary[str(n)]['return_levels'][distr], '-k')
                ax.fill_between(bm_summary[str(n)]['return_levels'].index.values, bm_summary[str(n)]['conf_int_low'][distr].values.astype(float), bm_summary[str(n)]['conf_int_high'][distr].values.astype(float), color='grey', edgecolor=None, alpha=0.1)
        if isinstance(obs_t[station], pd.Series):
                ax.plot(obs_t[station].index, obs_t[station].values, 'or')
        ax.set_xscale('log')
        ax.set_ylim(0, roundup(bm_summary[str(n)]['return_levels'][distr].max()))
        ax.set_xlabel('Return Period (year)')
        ax.set_ylabel('Discharge (m3/s)')
        plt.title(locs[station])
        plt.show()
        f.savefig(os.path.join(fn_fig, f'{station}_{distr}_return_curve_tradi.png'), dpi=400)

        #All years - 65 years
        n=1040
        f, ax = plt.subplots()
        ax.plot(bm_summary[str(n)]['return_levels'].index, bm_summary[str(n)]['return_levels']['gumbel_r'], '-b', label='Gumbel')
        ax.plot(bm_summary[str(n)]['return_levels'].index, bm_summary[str(n)]['return_levels']['genextreme'], '-g', label='GEV')
        ax.fill_between(bm_summary[str(n)]['return_levels'].index.values, bm_summary[str(n)]['conf_int_low']['gumbel_r'].values.astype(float), bm_summary[str(n)]['conf_int_high']['gumbel_r'].values.astype(float), color='blue', edgecolor=None, alpha=0.1)
        ax.fill_between(bm_summary[str(n)]['return_levels'].index.values, bm_summary[str(n)]['conf_int_low']['genextreme'].values.astype(float), bm_summary[str(n)]['conf_int_high']['genextreme'].values.astype(float), color='green', edgecolor=None, alpha=0.1)
        if isinstance(obs_t[station], pd.Series):
                ax.plot(obs_t[station].index, obs_t[station].values, 'or', label='reported')
        ax.set_xscale('log')
        ax.set_ylim(0, roundup(bm_summary[str(n)]['return_levels']['gumbel_r'].max()))
        ax.set_xlabel('Return Period (year)')
        ax.set_ylabel('Discharge (m3/s)')
        ax.legend()
        plt.title(locs[station] + f' - {n} years')
        plt.show()
        f.savefig(os.path.join(fn_fig, f'{station}_gumbel_gev_return_curve_{n}.png'), dpi=400)

        #Both - 65 years
        n=65
        f, ax = plt.subplots()
        ax.plot(bm_summary[str(n)]['return_levels'].index, bm_summary[str(n)]['return_levels']['gumbel_r'], '-b', label='Gumbel')
        ax.plot(bm_summary[str(n)]['return_levels'].index, bm_summary[str(n)]['return_levels']['genextreme'], '-g', label='GEV')
        ax.fill_between(bm_summary[str(n)]['return_levels'].index.values, bm_summary[str(n)]['conf_int_low']['gumbel_r'].values.astype(float), bm_summary[str(n)]['conf_int_high']['gumbel_r'].values.astype(float), color='blue', edgecolor=None, alpha=0.1)
        ax.fill_between(bm_summary[str(n)]['return_levels'].index.values, bm_summary[str(n)]['conf_int_low']['genextreme'].values.astype(float), bm_summary[str(n)]['conf_int_high']['genextreme'].values.astype(float), color='green', edgecolor=None, alpha=0.1)
        if isinstance(obs_t[station], pd.Series):
                ax.plot(obs_t[station].index, obs_t[station].values, 'or', label='reported')
        ax.set_xscale('log')
        ax.set_ylim(0, roundup(bm_summary[str(n)]['return_levels']['gumbel_r'].max()))
        ax.set_xlabel('Return Period (year)')
        ax.set_ylabel('Discharge (m3/s)')
        ax.legend()
        plt.title(locs[station] + f' - {n} years')
        plt.show()
        f.savefig(os.path.join(fn_fig, f'{station}_gumbel_gev_return_curve_{n}.png'), dpi=400)

        #
        T_considered = [10,50,100,500,1000]

        #Collecting the other way
        distr = 'genextreme'
        color=iter(cm.rainbow(np.linspace(0,1,len(T_considered)+1)))
        f, ax = plt.subplots()
        for T_i in T_considered:
                c=next(color)
                unc = pd.DataFrame(columns = bm_summary.keys(), index= ['ret_value', 'cil', 'cih'])
                for key in bm_summary.keys():
                        unc.loc['ret_value',key]=  bm_summary[key]['return_levels'].loc[T_i, distr]
                        unc.loc['cil',key]=  bm_summary[key]['conf_int_low'].loc[T_i, distr]
                        unc.loc['cih',key]=  bm_summary[key]['conf_int_high'].loc[T_i, distr]
                ax.plot([int(i) for i in unc.columns],unc.loc['ret_value',:],'o-', color=c, label = str(T_i))
                ax.fill_between([int(i) for i in unc.columns],unc.loc['cil',:].values.astype(float),unc.loc['cih',:].values.astype(float),color=c, alpha=0.15, edgecolor=None)
                if isinstance(obs_t[station], pd.Series):
                        try:
                                ax.plot([int(i) for i in unc.columns], np.repeat(obs_t[station].loc[float(T_i)], len([int(i) for i in unc.columns])), '--', color=c)
                        except:
                                continue
        ax.legend()
        ax.set_xlabel("Record length (years)")
        ax.set_ylabel("Discharge (m3/s)")
        plt.title(locs[station])
        #ax.set_ylim(2000, 6000)
        plt.show()
        f.savefig(os.path.join(fn_fig, f'{station}_{distr}_return_level_conv.png'), dpi=400)

        #Collecting the other way
        distr = 'gumbel_r'
        color=iter(cm.rainbow(np.linspace(0,1,len(T_considered)+1)))
        f, ax = plt.subplots()
        for T_i in T_considered:
                c=next(color)
                unc = pd.DataFrame(columns = bm_summary.keys(), index= ['ret_value', 'cil', 'cih'])
                for key in bm_summary.keys():
                        unc.loc['ret_value',key]=  bm_summary[key]['return_levels'].loc[T_i, distr]
                        unc.loc['cil',key]=  bm_summary[key]['conf_int_low'].loc[T_i, distr]
                        unc.loc['cih',key]=  bm_summary[key]['conf_int_high'].loc[T_i, distr]
                ax.plot([int(i) for i in unc.columns],unc.loc['ret_value',:],'o-', color=c, label = str(T_i))
                ax.fill_between([int(i) for i in unc.columns],unc.loc['cil',:].values.astype(float),unc.loc['cih',:].values.astype(float),color=c, alpha=0.15, edgecolor=None)
                # ax.plot([int(i) for i in unc.columns],unc.loc['cil',:],'--k')
                # ax.plot([int(i) for i in unc.columns],unc.loc['cih',:], '--k')
                if isinstance(obs_t[station], pd.Series):
                        try:
                                ax.plot([int(i) for i in unc.columns], np.repeat(obs_t[station].loc[float(T_i)], len([int(i) for i in unc.columns])), '--', color=c)
                        except:
                                continue
        ax.legend()
        ax.set_xlabel("Record length (years)")
        ax.set_ylabel("Discharge (m3/s)")
        plt.title(locs[station])
        #ax.set_ylim(2000, 6000)
        plt.show()
        f.savefig(os.path.join(fn_fig, f'{station}_{distr}_return_level_conv.png'), dpi=400)

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

#         #Parameter Stability:
#         ax = plot_mean_residual_life(ts_key)
#         fig = ax.get_figure()
#         plt.show()

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
