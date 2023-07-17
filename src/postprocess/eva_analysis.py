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


def calc_stats_emcee(model, Ts, distr="gumbel_r"):
        summ = pd.DataFrame(columns=[distr], index=['loc', 'scale', 'shape', 'AIC', 'rate']) 
        ret_levels = pd.DataFrame(columns=[distr], index=Ts) 
        cils = pd.DataFrame(columns=[distr], index=Ts) 
        cihs = pd.DataFrame(columns=[distr], index=Ts) 

        if distr == 'genextreme': #We correct the starting conditions from gumbel_r
                params_gumb = gumbel_r.fit(model.extremes.values)
                params_gev = genextreme.fit(model.extremes.values, loc = params_gumb[0], scale=params_gumb[1])
                model.fit_model(model = "Emcee", distribution = distr, distribution_kwargs={'floc': params_gev[1], 'fscale': params_gev[2]}) 
                summ.loc['loc', distr] = model.distribution.fixed_parameters['floc']
                summ.loc['scale', distr] = model.distribution.fixed_parameters['fscale']
        else:
                model.fit_model(model = "Emcee", distribution = distr) #try also Emcee at some point
                summ.loc['loc', distr] = model.distribution.mle_parameters['loc']
                summ.loc['scale', distr] = model.distribution.mle_parameters['scale']
        summ.loc['AIC', distr] = model.AIC
        if 'c' in model.distribution.mle_parameters.keys():
                summ.loc['shape', distr] = model.distribution.mle_parameters['c']
        summ.loc['rate', distr] = 1.0
        ret_lev, cil, cih = model.get_return_value(Ts, return_period_size = "365.2425D", alpha=0.95)
        ret_levels.loc[:,distr]=ret_lev
        cils.loc[:,distr]=cil
        cihs.loc[:,distr]=cih
        return summ, ret_levels, cils, cihs 
#%%
# We import the modelled data
Folder_start = "/p/11208719-interreg"
model_wflow = "f_spwgauges"
Folder_p = os.path.join(Folder_start, "wflow", model_wflow)
folder = "members_bias_corrected_daily" #"members_bias_corrected_hourly"
fn_fig = os.path.join(Folder_start, "Figures", model_wflow, folder)

dt = folder.split("_")[-1]
print(f"Performing analysis for {dt}")

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
        "Q_101": "Meuse at St-Mihiel" 

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
        "Q_101": "subcatch_S02",
        # Roer
        # Geul 
        } 
#%% We read the observed data

#SPW - official statistics
fn_spw = '/p/11208719-interreg/data/spw/statistiques/c_final/spw_statistics.nc' #spw_statistiques doesn't exist anymore
retlev_spw = xr.open_dataset(fn_spw)

#SPW - raw HOURLY data
fn_spw_raw = '/p/11208719-interreg/data/spw/Discharge/c_final/hourly_spw_discharges.nc' 
spw_hourly = xr.open_dataset(fn_spw_raw).load()

#For the official statistics sheet
locs_spw = {6228 : "Vesdre at Chaudfontaine",# SPW - old Q_12 - severely hit 2021
        6621 : "Ambleve at Martinrive",# SPW - old Q_11
        5921 : "Ourthe at Tabreux",# SPW - old Q_10
        8221 : "Lesse at Gendron", # SPW - old Q_801
#        "": "Sambre at Salzinnes", #SPW - old Q_9 - cannot find official statistics
        9434 : "Semois at Membre"}
#        "" : "Sambre at Floriffoux"} - cannot find ID in the nc file

#HYDRO
fn_hydro = '/p/11208719-interreg/data/hydroportail/c_final/hydro_statistiques_daily.nc'
retlev_hydro = xr.open_dataset(fn_hydro)
hydro_daily = xr.open_dataset('/p/11208719-interreg/data/hydroportail/c_final/hydro_daily.nc').load()

#We plot the locations?

#%%
#Official statistics
obs_t = {'Q_1011': retlev_hydro.isel(station=1).to_dataframe()[['valeur','int_conf_bas', 'int_conf_haut']], #hydrofrance
        'Q_4': retlev_hydro.isel(station=0).to_dataframe()[['valeur','int_conf_bas', 'int_conf_haut']], #hydrofrance et SPW
        'Q_16': 0, #/St Pieter -- station recalculée Discharge at Kanne + discharge at St Pieter 
        'Q_12': retlev_spw.isel(station=5).to_dataframe()[['valeur','int_conf_bas', 'int_conf_haut']],# SPW - severely hit 2021
        'Q_11': retlev_spw.isel(station=0).to_dataframe()[['valeur','int_conf_bas', 'int_conf_haut']],# SPW 
        'Q_10':  retlev_spw.isel(station=2).to_dataframe()[['valeur','int_conf_bas', 'int_conf_haut']],# SPW 
        'Q_801':  retlev_spw.isel(station=1).to_dataframe()[['valeur','int_conf_bas', 'int_conf_haut']], # SPW
        "Q_9": 0, #SPW
        'Q_101':0,
        # Roer
        # Geul 
        } 

#Original data - hourly or daily data
if dt == 'daily':
        data_t = {'Q_1011': hydro_daily['Q'].isel(wflow_id=0).to_series(), #hydrofrance - Meuse at Goncourt
                'Q_4': hydro_daily['Q'].isel(wflow_id=2).to_series(), #hydrofrance et SPW - "Meuse at Chooz"
                'Q_16': 0, #"Meuse at Borgharen"
                'Q_12': 0,# SPW - severely hit 2021 - "Vesdre at Chaudfontaine"
                'Q_11': 0,# SPW - "Ambleve at Martinrive"
                'Q_10':  0,# SPW - "Ourthe at Tabreux"
                'Q_801':  0, # SPW - "Lesse at Gendron"
                "Q_9": 0, #SPW - "Sambre at Salzinnes", 
                "Q_101": 0, # "Meuse at Saint Mihiel"
                # Roer
                # Geul 
                } 
if dt == 'hourly':
        data_t = {'Q_1011': 0, #hydrofrance - Meuse at Goncourt
                'Q_4': 0, #hydrofrance et SPW - "Meuse at Chooz"
                'Q_16': 0, #"Meuse at Borgharen"
                'Q_12': spw_hourly['Q'].sel(id=6228).to_series(),# SPW - severely hit 2021 - "Vesdre at Chaudfontaine"
                'Q_11': spw_hourly['Q'].sel(id=6621).to_series(),# SPW - "Ambleve at Martinrive"
                'Q_10':  spw_hourly['Q'].sel(id=5921).to_series(),# SPW - "Ourthe at Tabreux"
                'Q_801':  spw_hourly['Q'].sel(id=8221).to_series(), # SPW - "Lesse at Gendron"
                "Q_9": spw_hourly['Q'].sel(id=7319).to_series(), #SPW - "Sambre at Salzinnes", 
                "Q_101": 0, # "Meuse at Saint Mihiel"
                # Roer
                # Geul 
                } 

#We extract the 2021 value if present:
max_2021 = dict()
for loc in locs:
        try:
                max_2021[loc] = data_t[loc].loc['2021-06-01':'2021-08-01'].max()
        except:
                max_2021[loc] = None 
#%% We perform the statistics on the observations
data_Tcal = dict()
for station in locs.keys():
        print(station)
        #
        #station = 'Q_9' 
        #
        print(locs[station], station)
        if isinstance(data_t[station], pd.Series):
                print(f"Performing statistics on observed ts for {station}")
                #For now we select one and extract the extremes
                ts_peaks=pd.DataFrame()
                ts_key = data_t[station] #We need to remove years with full nans!!
                first_idx = ts_key.first_valid_index()
                last_idx = ts_key.last_valid_index()
                ts_key = ts_key.loc[first_idx:last_idx]

                #We perform the EVA - BM - Gumbel 
                peaks = pyex.get_extremes(ts_key, method = "BM", extremes_type="high", block_size="365.2426D", errors="ignore")
                ts_peaks = peaks.copy()

                emp_T = calculate_emp(ts_peaks.values)
                emp_month = ts_peaks.index.month

                data_Tcal[station] = dict()
                data_Tcal[station]['peaks'] = ts_peaks
                data_Tcal[station]['emp_T'] = emp_T
                data_Tcal[station]['emp_month'] = emp_month

                Ts = [1.1, 1.5, 1.8, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 1200]
                summ = pd.DataFrame(columns=['genextreme', 'gumbel_r', 'expon', 'genpareto'], index=['loc', 'scale', 'shape', 'AIC', 'rate']) 
                ret_levels = pd.DataFrame(columns=['genextreme', 'gumbel_r', 'expon', 'genpareto'], index=Ts) 
                cils = pd.DataFrame(columns=['genextreme', 'gumbel_r', 'expon', 'genpareto'], index=Ts) 
                cihs = pd.DataFrame(columns=['genextreme', 'gumbel_r', 'expon', 'genpareto'], index=Ts) 

                model = pyex.EVA.from_extremes(extremes = ts_peaks, method = "BM", extremes_type = 'high', block_size = "365.2425D")

                summ[['gumbel_r']], ret_levels[['gumbel_r']], cils[['gumbel_r']], cihs[['gumbel_r']] = calc_stats(model, Ts, distr="gumbel_r")
                summ[['genextreme']], ret_levels[['genextreme']], cils[['genextreme']], cihs[['genextreme']] = calc_stats(model, Ts, distr="genextreme")

                data_Tcal[station]['BM_fit'] = {'eva_params': summ, 'return_levels': ret_levels, 'conf_int_low': cils, 'conf_int_high': cihs}

                #We plot the time series
                f, ax = plt.subplots()
                data_t[station].dropna().plot(ax=ax, lw=0.5)
                ax.set_ylabel('Discharge ($m^3/s$)')
                ax.set_ylim(bottom=0)
                if isinstance(max_2021[station], float): #Observed time series
                        ax.hlines(y=max_2021[station], xmin=data_t[station].dropna().index[0], xmax=data_t[station].dropna().index[-1], color = 'chartreuse')
                plt.title(locs[station])
                f.savefig(os.path.join(fn_fig, f'{station}_{dt}_ts_observed.png'), dpi=400)

#%%
# Storing all the results in one location 
runs_dict = {}
for key in model_runs.keys():
    print(key)
    case = model_runs[key]["case"]
    ens = model_runs[key]["folder"] 
    runs_dict[key] = pd.read_csv(os.path.join(Folder_p, case, ens, "output.csv"), index_col=['time'], header=0, usecols = use_cols, parse_dates=['time'], date_parser = date_parser)
#%%
#station = 'Q_9'
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
for station in locs.keys():
        print(station)
#station = 'Q_101'
        #
        print(station)
        #For now we select one and extract the extremes
        ts_peaks=pd.DataFrame()
        ts_dates = pd.DataFrame()
        i = 0
        for key in model_runs.keys():
                print(i, key)
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
                
                print(len(ts_peaks))
                i += len(ts_key.index.year.unique())
        ts_peaks.rename(columns={0:station}, inplace = True)
        if len(ts_peaks) == 1040:
                print("Date conversion seems ok!")
        #ts_peaks.sort_index(axis = 0, ascending=True, inplace = True)

        emp_T = calculate_emp(ts_peaks.values)
        emp_n1 = calculate_emp(ts_peaks.iloc[0:65].values) #We have 65 years
        emp_month = ts_dates['month']

        #Plotting the histogram
        top_n = [10, 50, 104, 500, 1040]
        for n in top_n:
                #PLotting the histogram of the highest values
                fig = plt.figure()
                #I shoudl use ax.bar instead()
                plt.hist(ts_dates.loc[emp_T.sort_values(by='rank', ascending = True).iloc[0:n].index]['month'], bins=np.arange(14)-0.5, edgecolor = 'black', color='blue', stacked = True, density = True)
                plt.xticks(range(13))
                plt.title(f'Top {n} AM events - {locs[station]}')
                plt.xlim(0.5,12.5)
                plt.show()
                fig.savefig(os.path.join(fn_fig, f'{station}_{dt}_top_hist_{n}_AM_events.png'), dpi=400)

        #We sample some data length and calculate the statistics 
        Ts = [1.1, 1.5, 1.8, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 1200]
        bm_summary = dict()

        #Doing it per ensemble for genextreme and gumbel_r
        i_start = 0
        j = 1
        for n in np.arange(65, 1040+65, 65):
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
        #
        #Doing it per concatenated ensemble for genextreme and gumbel_r
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

                if (station == 'Q_101') and (dt == 'daily'): #and (n == 1040):
                        summ[['genextreme']], ret_levels[['genextreme']], cils[['genextreme']], cihs[['genextreme']] = calc_stats_emcee(model, Ts, distr="genextreme")

                bm_summary[str(n)] = {'eva_params': summ, 'return_levels': ret_levels, 'conf_int_low': cils, 'conf_int_high': cihs}

        #
        #For plotting
        max_y = roundup(max(bm_summary[str(1040)]['return_levels']['gumbel_r'].max(), bm_summary[str(1040)]['return_levels']['genextreme'].max(), emp_T.value.max(), bm_summary[str(1040)]['conf_int_high']['gumbel_r'].max(), bm_summary[str(1040)]['conf_int_high']['genextreme'].max()))

        #Plotting results - return levels per 65 years ensemble
        f, ax = plt.subplots()
        distr = 'gumbel_r'
        for n in np.arange(1, 17, 1):
                print(n)
                ax.plot(bm_summary[f"65_r{n}"]['return_levels'].index, bm_summary[f"65_r{n}"]['return_levels'][distr], '-k', zorder = 2)
                ax.fill_between(bm_summary[f"65_r{n}"]['return_levels'].index.values, bm_summary[f"65_r{n}"]['conf_int_low'][distr].values.astype(float), bm_summary[f"65_r{n}"]['conf_int_high'][distr].values.astype(float), color='grey', edgecolor=None, alpha=0.1, zorder = 1)
        if isinstance(obs_t[station], pd.DataFrame): #Official statistics
                ax.plot(obs_t[station].index, obs_t[station]['valeur'].values, 'or', zorder = 5, label = 'official stat.')
                ax.vlines(x= obs_t[station].index, ymin = obs_t[station]['int_conf_bas'].values, ymax = obs_t[station]['int_conf_haut'].values, color='r', zorder = 5)
        if isinstance(data_t[station], pd.Series): #Observed time series
                ax.scatter(data_Tcal[station]['emp_T'].emp_rp.values, data_Tcal[station]['emp_T'].value, c = data_Tcal[station]['emp_month'].values, marker = 'x', label='observed', cmap = cmap, norm=norm, zorder = 7)
        if isinstance(max_2021[station], float): #Observed 2021 level
                ax.hlines(y=max_2021[station], xmin=min(Ts), xmax=max(Ts), color = 'chartreuse', label = '2021 level')
        m = ax.scatter(emp_T.emp_rp.values, emp_T.value, c = emp_month.values, marker = '.', label='modelled', cmap = cmap, norm=norm, zorder = 6)        
        ax.plot(bm_summary[str(1040)]['return_levels'].index, bm_summary[str(1040)]['return_levels'][distr], '-r', label='Gumbel', zorder = 4)
        ax.fill_between(bm_summary[str(1040)]['return_levels'].index.values, bm_summary[str(1040)]['conf_int_low'][distr].values.astype(float), bm_summary[str(1040)]['conf_int_high'][distr].values.astype(float), color='darkred', edgecolor=None, alpha=0.5, zorder = 3)        
        ax.set_xscale('log')
        ax.set_ylim(0, max_y)
        ax.set_xlim(1.0,1200)
        ax.set_xlabel('Return Period (year)')
        ax.set_ylabel('Discharge ($m^3/s$)')
        ax.legend()
        plt.title(locs[station])
        plt.show()
        f.savefig(os.path.join(fn_fig, f'{station}_{dt}_{distr}_return_curve_tradi.png'), dpi=400)

        f, ax = plt.subplots()
        distr = 'genextreme'
        for n in np.arange(1, 17, 1):
                print(n)
                ax.plot(bm_summary[f"65_r{n}"]['return_levels'].index, bm_summary[f"65_r{n}"]['return_levels'][distr], '-k', zorder = 2)
                ax.fill_between(bm_summary[f"65_r{n}"]['return_levels'].index.values, bm_summary[f"65_r{n}"]['conf_int_low'][distr].values.astype(float), bm_summary[f"65_r{n}"]['conf_int_high'][distr].values.astype(float), color='grey', edgecolor=None, alpha=0.1, zorder = 1)
        if isinstance(obs_t[station], pd.DataFrame): #Official statistics
                ax.plot(obs_t[station].index, obs_t[station]['valeur'].values, 'or', zorder = 5, label = 'official stat.')
                ax.vlines(x= obs_t[station].index, ymin = obs_t[station]['int_conf_bas'].values, ymax = obs_t[station]['int_conf_haut'].values, color='r', zorder = 5)
        if isinstance(data_t[station], pd.Series): #Observed time series
                ax.scatter(data_Tcal[station]['emp_T'].emp_rp.values, data_Tcal[station]['emp_T'].value, c = data_Tcal[station]['emp_month'].values, marker = 'x', label='observed', cmap = cmap, norm=norm, zorder = 7)
        if isinstance(max_2021[station], float): #Observed time series
                ax.hlines(y=max_2021[station], xmin=min(Ts), xmax=max(Ts), color = 'chartreuse', label = '2021 level')        
        m = ax.scatter(emp_T.emp_rp.values, emp_T.value, c = emp_month.values, marker = '.', label='modelled', cmap = cmap, norm=norm, zorder = 6)        
        ax.plot(bm_summary[str(1040)]['return_levels'].index, bm_summary[str(1040)]['return_levels'][distr], '-r', label='GEV', zorder = 4)
        ax.fill_between(bm_summary[str(1040)]['return_levels'].index.values, bm_summary[str(1040)]['conf_int_low'][distr].values.astype(float), bm_summary[str(1040)]['conf_int_high'][distr].values.astype(float), color='darkred', edgecolor=None, alpha=0.5, zorder = 3)        
        ax.set_xscale('log')
        ax.set_ylim(0, max_y)
        ax.set_xlim(1.0,1200)
        ax.set_xlabel('Return Period (year)')
        ax.set_ylabel('Discharge ($m^3/s$)')
        ax.legend()
        plt.title(locs[station])
        plt.show()
        f.savefig(os.path.join(fn_fig, f'{station}_{dt}_{distr}_return_curve_tradi.png'), dpi=400)

        #All years - 1040 years
        n=1040
        f, ax = plt.subplots()
        ax.plot(bm_summary[str(n)]['return_levels'].index, bm_summary[str(n)]['return_levels']['gumbel_r'], '-b', label='Gumbel')
        ax.plot(bm_summary[str(n)]['return_levels'].index, bm_summary[str(n)]['return_levels']['genextreme'], '-g', label='GEV')
        ax.fill_between(bm_summary[str(n)]['return_levels'].index.values, bm_summary[str(n)]['conf_int_low']['gumbel_r'].values.astype(float), bm_summary[str(n)]['conf_int_high']['gumbel_r'].values.astype(float), color='blue', edgecolor=None, alpha=0.1)
        ax.fill_between(bm_summary[str(n)]['return_levels'].index.values, bm_summary[str(n)]['conf_int_low']['genextreme'].values.astype(float), bm_summary[str(n)]['conf_int_high']['genextreme'].values.astype(float), color='green', edgecolor=None, alpha=0.1)
        m = ax.scatter(emp_T.emp_rp.values, emp_T.value, c = emp_month.values, marker = '.', label='modelled', cmap = cmap, norm=norm)
        if isinstance(data_t[station], pd.Series):
                ax.scatter(data_Tcal[station]['emp_T'].emp_rp.values, data_Tcal[station]['emp_T'].value, c = data_Tcal[station]['emp_month'].values, marker = 'x', label='observed', cmap = cmap, norm=norm)
        if isinstance(obs_t[station], pd.DataFrame):
                ax.plot(obs_t[station].index, obs_t[station]['valeur'].values, 'or', label='official statistics')
                ax.vlines(x= obs_t[station].index, ymin = obs_t[station]['int_conf_bas'].values, ymax = obs_t[station]['int_conf_haut'].values, color='r')
        if isinstance(max_2021[station], float): #Observed time series
                ax.hlines(y=max_2021[station], xmin=min(Ts), xmax=max(Ts), color = 'chartreuse', label = '2021 level')        
        ax.set_xscale('log')
        ax.set_ylim(0, max_y)
        ax.set_xlim(1.0,1200)
        ax.set_xlabel('Return Period (year)')
        ax.set_ylabel('Discharge (m3/s)')
        # produce a legend with the unique colors from the scatter
        legend1 = ax.legend(*m.legend_elements(), title="Month", loc="upper left", fontsize = 7)
        #ax.add_artist(legend1)
        ax.legend()
        plt.title(locs[station] + f' - {n} years')
        plt.show()
        f.savefig(os.path.join(fn_fig, f'{station}_{dt}_gumbel_gev_return_curve_{n}.png'), dpi=400)

        #Both - 65 years
        n=65
        f, ax = plt.subplots()
        ax.plot(bm_summary[str(n)]['return_levels'].index, bm_summary[str(n)]['return_levels']['gumbel_r'], '-b', label='Gumbel')
        ax.plot(bm_summary[str(n)]['return_levels'].index, bm_summary[str(n)]['return_levels']['genextreme'], '-g', label='GEV')
        ax.fill_between(bm_summary[str(n)]['return_levels'].index.values, bm_summary[str(n)]['conf_int_low']['gumbel_r'].values.astype(float), bm_summary[str(n)]['conf_int_high']['gumbel_r'].values.astype(float), color='blue', edgecolor=None, alpha=0.1)
        ax.fill_between(bm_summary[str(n)]['return_levels'].index.values, bm_summary[str(n)]['conf_int_low']['genextreme'].values.astype(float), bm_summary[str(n)]['conf_int_high']['genextreme'].values.astype(float), color='green', edgecolor=None, alpha=0.1)
        m = ax.scatter(emp_n1.emp_rp.values, emp_n1.value, c = emp_month[0:65].values, marker = '.', label='modelled', cmap = cmap, norm=norm)
        if isinstance(data_t[station], pd.Series):
                ax.scatter(data_Tcal[station]['emp_T'].emp_rp.values, data_Tcal[station]['emp_T'].value, c = data_Tcal[station]['emp_month'].values, marker = 'x', label='observed', cmap = cmap, norm=norm)
        if isinstance(obs_t[station], pd.DataFrame):
                ax.plot(obs_t[station].index, obs_t[station]['valeur'].values, 'or', label='official statistics')
                ax.vlines(x= obs_t[station].index, ymin = obs_t[station]['int_conf_bas'].values, ymax = obs_t[station]['int_conf_haut'].values, color='r')
        if isinstance(max_2021[station], float): #Observed time series
                ax.hlines(y=max_2021[station], xmin=min(Ts), xmax=max(Ts), color = 'chartreuse', label = '2021 level')        
        ax.set_xscale('log')
        ax.set_ylim(0, max_y)
        ax.set_xlim(1.0,1200)
        ax.set_xlabel('Return Period (year)')
        ax.set_ylabel('Discharge (m3/s)')
        legend1 = ax.legend(*m.legend_elements(), title="Month", loc="upper left", fontsize = 7)
        #ax.add_artist(legend1)
        ax.legend()
        plt.title(locs[station] + f' - {n} years')
        plt.show()
        f.savefig(os.path.join(fn_fig, f'{station}_{dt}_gumbel_gev_return_curve_{n}.png'), dpi=400)

        #
        T_considered = [10,50,100,500,1000]
        cols = [ str(n) for n in np.arange(65, 1040+65, 65)]

        #Collecting the other way
        distr = 'genextreme'
        color=iter(cm.rainbow(np.linspace(0,1,len(T_considered)+1)))
        f, ax = plt.subplots()
        for T_i in T_considered:
                c=next(color)
                unc = pd.DataFrame(columns = cols, index= ['ret_value', 'cil', 'cih'])
                for key in cols:
                        unc.loc['ret_value',key]=  bm_summary[key]['return_levels'].loc[T_i, distr]
                        unc.loc['cil',key]=  bm_summary[key]['conf_int_low'].loc[T_i, distr]
                        unc.loc['cih',key]=  bm_summary[key]['conf_int_high'].loc[T_i, distr]
                ax.plot([int(i) for i in unc.columns],unc.loc['ret_value',:],'o-', color=c, label = str(T_i))
                ax.fill_between([int(i) for i in unc.columns],unc.loc['cil',:].values.astype(float),unc.loc['cih',:].values.astype(float),color=c, alpha=0.15, edgecolor=None)
                if isinstance(obs_t[station], pd.DataFrame):
                        try:
                                ax.plot([int(i) for i in unc.columns], np.repeat(obs_t[station]['valeur'].loc[float(T_i)], len([int(i) for i in unc.columns])), '--', color=c)
                        except:
                                continue
        ax.legend()
        ax.set_xlabel("Record length (years)")
        ax.set_ylabel("Discharge (m3/s)")
        plt.title(locs[station]+' - '+ distr)
        #ax.set_ylim(2000, 6000)
        plt.show()
        f.savefig(os.path.join(fn_fig, f'{station}_{dt}_{distr}_return_level_conv.png'), dpi=400)

        #Collecting the other way
        distr = 'gumbel_r'
        color=iter(cm.rainbow(np.linspace(0,1,len(T_considered)+1)))
        f, ax = plt.subplots()
        for T_i in T_considered:
                c=next(color)
                unc = pd.DataFrame(columns = cols, index= ['ret_value', 'cil', 'cih'])
                for key in cols:
                        unc.loc['ret_value',key]=  bm_summary[key]['return_levels'].loc[T_i, distr]
                        unc.loc['cil',key]=  bm_summary[key]['conf_int_low'].loc[T_i, distr]
                        unc.loc['cih',key]=  bm_summary[key]['conf_int_high'].loc[T_i, distr]
                ax.plot([int(i) for i in unc.columns],unc.loc['ret_value',:],'o-', color=c, label = str(T_i))
                ax.fill_between([int(i) for i in unc.columns],unc.loc['cil',:].values.astype(float),unc.loc['cih',:].values.astype(float),color=c, alpha=0.15, edgecolor=None)
                # ax.plot([int(i) for i in unc.columns],unc.loc['cil',:],'--k')
                # ax.plot([int(i) for i in unc.columns],unc.loc['cih',:], '--k')
                if isinstance(obs_t[station], pd.DataFrame):
                        try:
                                ax.plot([int(i) for i in unc.columns], np.repeat(obs_t[station]['valeur'].loc[float(T_i)], len([int(i) for i in unc.columns])), '--', color=c)
                        except:
                                continue
        ax.legend()
        ax.set_xlabel("Record length (years)")
        ax.set_ylabel("Discharge (m3/s)")
        plt.title(locs[station]+' - '+distr)
        #ax.set_ylim(2000, 6000)
        plt.show()
        f.savefig(os.path.join(fn_fig, f'{station}_{dt}_{distr}_return_level_conv.png'), dpi=400)

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
