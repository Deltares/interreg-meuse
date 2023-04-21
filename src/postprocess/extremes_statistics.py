#%%
import pandas as pd
import glob
import xarray as xr
from datetime import datetime
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import matplotlib as mpl


#%% Import the results
fn_res = r'p:\11208719-interreg\wflow\d_manualcalib\members_bias_corrected_daily'
fn_fig = r'p:\11208719-interreg\Figures\members_bias_corrected_daily'
fn_runs = glob.glob(os.path.join(fn_res,'*','output.csv'))
date_parser = lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S")

#Important locations
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

#%%
Folder_p = r"p:\11208719-interreg\wflow\d_manualcalib"

model_runs = {
    "Obs.": {"case":"members_bias_corrected_daily", 
            "folder": "r1i1p5f1"},
    
    "r2": {"case":"members_bias_corrected_daily", 
            "folder": "r2i1p5f1"},
    "r3": {"case":"members_bias_corrected_daily", 
            "folder": "r3i1p5f1"},
    "r4": {"case":"members_bias_corrected_daily", 
            "folder": "r4i1p5f1"}, 
    "r5": {"case":"members_bias_corrected_daily", 
            "folder": "r5i1p5f1"}, 
    "r6": {"case":"members_bias_corrected_daily", 
            "folder": "r6i1p5f1"}, 
    "r7": {"case":"members_bias_corrected_daily", 
            "folder": "r7i1p5f1"}, 
    "r8": {"case":"members_bias_corrected_daily", 
            "folder": "r8i1p5f1"}, 
    "r9": {"case":"members_bias_corrected_daily", 
            "folder": "r9i1p5f1"}, 
    "r10": {"case":"members_bias_corrected_daily", 
            "folder": "r10i1p5f1"}, 
    "r11": {"case":"members_bias_corrected_daily", 
            "folder": "r11i1p5f1"}, 
    "r12": {"case":"members_bias_corrected_daily", 
            "folder": "r12i1p5f1"}, 
    "r13": {"case":"members_bias_corrected_daily", 
            "folder": "r13i1p5f1"}, 
    "r14": {"case":"members_bias_corrected_daily", 
            "folder": "r14i1p5f1"}, 
    "r15": {"case":"members_bias_corrected_daily", 
            "folder": "r15i1p5f1"}, 
    "r16": {"case":"members_bias_corrected_daily", 
            "folder": "r16i1p5f1"}, 
}


### prepare dataset to make plots
colors = [
    '#a6cee3','#1f78b4',
    '#b2df8a','#33a02c',
    '#fb9a99','#e31a1c',
    '#fdbf6f','#ff7f00',
    '#cab2d6','#6a3d9a',
    '#ffff99','#b15928']

runs_dict = {}

for key in model_runs.keys():
    print(key)
    case = model_runs[key]["case"]
    folder = model_runs[key]["folder"] 
    runs_dict[key] = pd.read_csv(os.path.join(Folder_p, case, folder, "output.csv"), index_col=['time'], header=0, usecols = use_cols, parse_dates=['time'], date_parser = date_parser)

plot_colors = colors[:len(runs_dict)]


#%% Reading the data
fn = fn_runs[0]
use_cols = ['time'] + list(locs.keys()) 

#df = pd.read_csv(fn, index_col=['time'], parse_dates=['time'], date_parser = date_parser)

#%%Subselection
cols = list(locs.keys()) 
ams_dict = {}

for key in runs_dict.keys():
    print(key)
    qs = runs_dict[key].loc[:,cols]

    #We extract the month of the maxima
    ams = pd.DataFrame()
    for r in cols:
        print(r)
        ams_vals = qs[r].groupby(pd.Grouper(freq="A-SEP")).max()
        ams_dates = qs[r].groupby(pd.Grouper(freq="A-SEP")).idxmax()
        ams[f'{r}_am'] = ams_vals
        ams[f'{r}_date'] = ams_dates
        ams[f'{r}_month'] = ams_dates.dt.month
    
    ams_dict[key] = ams


#%% We plot the time series
for key in runs_dict.keys():
    print(key) 
    n= len(cols)
    #color=iter(cm.rainbow(np.linspace(0,1,n)))

    cmap = mpl.cm.hsv
    norm = mpl.colors.Normalize(vmin=1, vmax=12)

    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(12, 20), squeeze=0)
    axs = axes.reshape(-1)
    i=0
    for r in cols:
        print(r)
        # c=next(color)
        v = axs[i].scatter(ams_dict[key][f'{r}_date'], ams_dict[key][f'{r}_am'], c=ams_dict[key][f'{r}_month'], cmap = cmap, norm=norm, label=r, zorder = 1)
        axs[i].plot(runs_dict[key].index, runs_dict[key].loc[:,r].values, c='k', label=r, zorder = 2)
        axs[i].set_ylabel('Q (m3/s) - {}'.format(r))
        #ax.set_xlim([datetime.date(2009, 1, 1), datetime.date(2009, 12, 31)])
        i += 1   
    fig.colorbar(v,orientation='horizontal', label='month')
    fig.savefig(os.path.join(fn_fig, f'ams_{key}.png'), dpi=400)
    plt.close()

#%% Plotting month of extreme as an histogram
for key in runs_dict.keys():
    print(key) 
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(12, 20), squeeze=0)
    axs = axes.reshape(-1)
    i=0
    for r in cols:
        print(r)
        axs[i].hist(ams_dict[key][f'{r}_month'], bins=np.arange(14)-0.5, edgecolor = 'black', color='blue')
        axs[i].set_xticks(range(13))
        axs[i].set_ylabel('Count - {}'.format(r))
        i += 1
    fig.savefig(os.path.join(fn_fig, f'ams_month_{key}.png'), dpi=400)
    plt.close()
#%% Gumbel plot
#gumbel high axes[6]
a=0.3
b = 1.-2.*a
for key in runs_dict.keys():
    print(key) 
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(12, 20), squeeze=0)
    axs = axes.reshape(-1)
    i=0

    fs=8
    for r in cols:
        print(r)
        ymin, ymax = 0, np.ceil(ams_dict[key][f'{r}_am'].max())
        p1 = ((np.arange(1,len(ams_dict[key][f'{r}_am'])+1.)-a))/(len(ams_dict[key][f'{r}_am'])+b)
        RP1 = 1/(1-p1)
        gumbel_p1 = -np.log(-np.log(1.-1./RP1))
        ts = [2., 5.,10.,30.] #,30.,100.,300.,1000.,3000.,10000.,30000.]
        #plot
        axs[i].plot(gumbel_p1, ams_dict[key].loc[:,f'{r}_am'].sort_values(), marker = '+', color = 'k', linestyle = 'None', label = 'Obs.', markersize = 6)
        # for label, color in zip(labels, colors):
        #     axes[6].plot(gumbel_p1, dsq_max['Q'].sel(runs = label).sortby(dsq_max['Q'].sel(runs = label)), marker = 'o', color = color, linestyle = 'None', label = label, markersize = 4)
        # axes[6].plot(gumbel_p1, dsq_max['Q'].sel(runs = label_01).sortby(dsq_max['Q'].sel(runs = label_01)), marker = '.', color = color_01, linestyle = 'None', label = label_01, markersize = 3)

        for t in ts:
            axs[i].axvline(-np.log(-np.log(1-1./t)),c='0.5', alpha=0.4)
            axs[i].text(-np.log(-np.log(1-1./t)),ymax*1.05,f'{t:.0f}y', fontsize = fs, ha="center", va="bottom")

        axs[i].set_ylabel('max. annual Q (m$^3$s$^{-1}$)', fontsize = fs)
        axs[i].set_xlabel('Plotting position and associated return period', fontsize = fs)
        i+=1
    fig.savefig(os.path.join(fn_fig, f'gumbel_plot_{key}.png'), dpi=400)
    plt.close()
#%% We look at the ranking and return period from the

from scipy.stats import gumbel_r
def calculate_emp(data):
    emp_p = data
    #emp_p = pd.DataFrame(data=data, columns=col)
    emp_p['rank'] = emp_p.iloc[:,0].rank(axis=0, ascending=False, method = 'dense')
    emp_p['exc_prob'] = emp_p['rank']/(emp_p['rank'].size+1) #change this line with what Anaïs sends to me, but is already correct
    emp_p['cum_prob'] = 1 - emp_p['exc_prob']
    emp_p['emp_rp'] = 1/emp_p['exc_prob']
    return emp_p

ens_nb = [f'{i}_ens' for i in np.arange(1,17,1)]

#We create the datasets based on the reading 
# df = pd.DataFrame()
# for key, ens in zip(model_runs.keys(), ens_nb):
#     df_i = runs_dict[key]
#     #We shift the time
#     start =
#     df = pd
#     runs_dict[ens] = 


T_ens = pd.DataFrame(columns=ens_nb)
T_50ens = pd.DataFrame(columns=ens_nb)
T_100ens = pd.DataFrame(columns=ens_nb)
T_500ens = pd.DataFrame(columns=ens_nb)
T_1000ens = pd.DataFrame(columns=ens_nb)

df_ens=pd.DataFrame()
for nb, key in zip(ens_nb, ams_dict.keys()):
    print(nb,key)
    df_i = ams_dict[key].reset_index(drop=True)
    df_ens = pd.concat([df_i, df_ens], axis = 0, ignore_index=True)

    #We do the ranking and the fit
    col = "Q_9"
    rank = calculate_emp(df_ens[[f'{col}_am']])

    #We fit gumbel
    model = gumbel_r.fit(df_ens[[f'{col}_am']].values)
    T = np.array([5,50,100,500,1000])
    p_T = 1/T
    p_inv= np.linspace(0.00001,0.999,100)
    ret_values = gumbel_r.isf(p_T, loc=model[0], scale = model[1])
    T_ens.loc[:,nb] = ret_values
    print(nb, ret_values)
    print("len of data", len(df_ens[[f'{col}_am']]))

    plt.figure()
    plt.plot(rank['emp_rp'], rank[f'{col}_am'], 'ok')
    plt.plot(1/p_inv, gumbel_r.isf(p_inv, loc=model[0], scale = model[1]), '-k')
    plt.ylabel(col)
    plt.xscale('log')
    #plt.ylim([0,1500])
    plt.xlim([1,2000])

    plt.savefig(os.path.join(fn_fig, f'{col}_{nb}.png'), dpi = 400)
    plt.close()

T_ens = T_ens.set_index(T)
plt.figure()
for t in T_ens.index:
    plt.plot(np.arange(1,17,1), T_ens.loc[t,:], 'o', label=t)
plt.ylabel(f'{col}')
plt.xlabel('nb. of Ensemble')
plt.legend()
plt.savefig(os.path.join(fn_fig, f'{col}_T.png'), dpi = 400)













#%%
import os
from func_plot_signature_joost import plot_hydro
from func_plot_signature_joost import plot_signatures

#%%
stations = cols
caserun = "r2"

Folder_plots = r"p:\11208719-interreg\Figures\members_bias_corrected_daily" + "\\" + f"{caserun}"

if not os.path.exists(Folder_plots):
    os.mkdir(Folder_plots)

#make dataset
variables = ['Q','P', 'EP']    
    
start = '1950-01-01'
end = '2014-12-31'
rng = pd.date_range(start, end, freq="D")

S = np.zeros((len(rng), len(cols), len(list(runs_dict.keys()))))
v = (('time', 'stations', 'runs'), S)
h = {k:v for k in variables}

ds = xr.Dataset(
        data_vars=h, 
        coords={'time': rng,
                'stations': stations,
                'runs': list(list(runs_dict.keys()))})
ds = ds * np.nan

for key in runs_dict.keys(): 
    print(key)
    #fill dataset with model runs
    #ds['Q'].loc[dict(runs = key)] = runs_dict[key][['Q_' + sub for sub in list(map(str,[1011,4,16,12,11,10,801,9]))]].loc[start:end]  
    ds['Q'].loc[dict(runs = key)] = runs_dict[key][cols].loc[start:end]    
#%%
#make plots
start_long = start
end_long =  end
start_1 =  '2010-11-01'
end_1 = '2011-03-01'
start_2 =  '2011-03-01'
end_2 =  '2011-10-31'
start_3 =  '2015-01-01'
end_3 = '2015-12-31'

for station_id, station_name in locs.items():
    # if station_id in qobs_h.stations.values:    
    print(station_name)
    runs_sel = list(runs_dict.keys()) 
    plot_colors = colors[:len(runs_dict)]
    #dsq = ds.sel(stations = station_id).sel(time = slice(start, end), runs=runs_sel)#.dropna(dim='time')
    #plot hydro
    #plot_hydro(dsq, start_long, end_long, start_1, end_1, start_2, end_2, start_3, end_3, runs_sel, plot_colors, Folder_plots, station_name, save=True)
    #plt.close()
        
    #make plot using function
    #dropna for signature calculations. 
    #start later for for warming up
    dsq = ds['Q'].sel(stations = station_id).sel(time = slice('1950-01-01', "2014-12-31"), runs=runs_sel).to_dataset().dropna(dim='time')
    #TODO: somehow xr.infer_freq(dsq.time) does not work for Borgharen..... 
    plot_signatures(dsq, runs_sel, plot_colors, Folder_plots, station_name, save=True, window=7*24)
    plt.close()
else:
    print(f"no obs data for {station_name}")

#%%Observations - p:\11208719-interreg\data\spw\statistiques
obs_stats = xr.open_dataset(r'p:\11208719-interreg\data\spw\statistiques\c_final\spw_statistiques.nc')
# %%
