#%%Import packages
import xarray as xr
import xclim
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r, genextreme
import hydromt
from hydromt_wflow import WflowModel
import numpy as np
import math

def roundup(x):
    return math.ceil(x / 10.0) * 10

def rounddown(x):
    return math.floor(x / 10.0) * 10

def calculate_emp(data):
    emp_p = pd.DataFrame(data=data, columns=['value'])
    emp_p['rank'] = emp_p.iloc[:,0].rank(axis=0, ascending=False, method = 'dense')
    emp_p['exc_prob'] = (emp_p['rank']-0.3)/(emp_p['rank'].size+0.4) #change this line with what Anaïs sends to me, but is already correct
    emp_p['cum_prob'] = 1 - emp_p['exc_prob']
    emp_p['emp_rp'] = 1/emp_p['exc_prob']
    return emp_p

# We calculate the Gumbel and GEV fit
def fit_gumb_gev(data, Ts):
    summ = pd.DataFrame(columns=['gumbel_r','genextreme'], index=['loc', 'scale', 'shape', 'AIC', 'rate']) 
    ret_levels = pd.DataFrame(columns=['gumbel_r','genextreme'], index=Ts) 

    params_gumb = gumbel_r.fit(data)
    params_gev = genextreme.fit(data, loc = params_gumb[0], scale=params_gumb[1])
    #params_gev = genextreme.fit(data) 

    summ.loc['loc', 'gumbel_r'] = params_gumb[0]
    summ.loc['scale', 'gumbel_r'] = params_gumb[1]
    summ.loc['loc', 'genextreme'] = params_gev[1]
    summ.loc['scale', 'genextreme'] = params_gev[2]
    summ.loc['shape', 'genextreme'] = params_gev[0]

    ret_levels.loc[Ts, 'gumbel_r'] = gumbel_r.isf(1/Ts, loc=summ.loc['loc', 'gumbel_r'], scale=summ.loc['scale', 'gumbel_r'])
    ret_levels.loc[Ts, 'genextreme'] = genextreme.isf(1/Ts, summ.loc['shape', 'genextreme'], loc=summ.loc['loc', 'genextreme'], scale=summ.loc['scale', 'genextreme'])

    return summ, ret_levels

def combine_probs_gumbel(quantiles, summ_w, summ_s, distr='gumbel_r'):
    df_q = pd.DataFrame(index=quantiles, columns=[['winter','summer', 'sum']])
    df_q['winter'] = gumbel_r.sf(quantiles, loc=summ_w.loc['loc', 'gumbel_r'], scale=summ_w.loc['scale', 'gumbel_r'])
    df_q['summer'] = gumbel_r.sf(quantiles, loc=summ_s.loc['loc', 'gumbel_r'], scale=summ_s.loc['scale', 'gumbel_r'])
    df_q['exc_prob'] = df_q.loc[:,'winter'].values + df_q.loc[:,'summer'].values
    return df_q

def combine_probs_gev(quantiles, summ_w, summ_s, distr='genextreme'):
    df_q = pd.DataFrame(index=quantiles, columns=[['winter','summer', 'sum']])
    df_q['winter'] = genextreme.sf(quantiles, summ_w.loc['shape', 'genextreme'], loc=summ_w.loc['loc', 'genextreme'], scale=summ_w.loc['scale', 'genextreme'])
    df_q['summer'] = genextreme.sf(quantiles, summ_s.loc['shape', 'genextreme'], loc=summ_s.loc['loc', 'genextreme'], scale=summ_s.loc['scale', 'genextreme'])
    df_q['exc_prob'] = df_q.loc[:,'winter'].values + df_q.loc[:,'summer'].values
    return df_q

#%% Set appropriate locations
Folder_start = r"/p/11208719-interreg"
model_wflow = "p_geulrur"
Folder_p = os.path.join(Folder_start, "wflow", model_wflow)
folder = "members_bias_corrected_revised_daily"
fn_fig = os.path.join(Folder_start, "Figures", model_wflow, folder)
fn_data_out = os.path.join(fn_fig, 'data')

#%% read one model 
# root = r"p:/11208719-interreg/wflow/o_rwsinfo/run_rwsinfo"
# folder = 'members_bias_corrected_revised_daily'
# config_fn = r"p:\11208719-interreg\wflow\o_rwsinfo\run_rwsinfo\run_rwsinfo.toml"
# yml = r"p:/11208719-interreg/data/data_meuse.yml"
# mod = WflowModel(root = root, config_fn=config_fn, data_libs=["deltares_data", yml], mode = "r")

# mod.staticgeoms.keys()

# mod.staticgeoms["gauges_rwsinfo"]
# mod.staticgeoms["gauges_waterschaplimburg"] #Geul at Meerssen

#%% We open some data
summer = xr.open_dataset(os.path.join(fn_data_out,'AM_summer_4_9_Oct.nc'))
winter = xr.open_dataset(os.path.join(fn_data_out,'AM_winter_10_3_Oct.nc'))

#%%
locs = {'Geul at Meersen': {'lat': 50.89167, 'lon': 5.72750, 'id': 1036},
        'Geul at Hommerich': {'lat': 50.80000, 'lon': 5.91917, 'id': 1030},
        'Sambre at Salzinne': {'lat': 50.45833, 'lon': 4.83583, 'id': 7319},
        'Ourthe at Tabreux': {'lat': 50.44167, 'lon': 5.53583, 'id': 5921},
        'Vesdre at Chaudfontaine': {'lat': 50.59167, 'lon': 5.65250, 'id': 6228},
        'Ambleve at Martinrive': {'lat': 50.48333, 'lon': 5.63583, 'id': 6621},
        'Semois at Membre Pont': {'lat': 49.86667, 'lon': 4.90250, 'id': 9434},
        'Viroin Treignes': {'lat': 50.09167, 'lon': 4.67750, 'id': 9021},
        'Lesse at Gendron': {'lat': 50.20833, 'lon': 4.96083, 'id': 8221},
        'Meuse at Goncourt': {'lat': 48.24167, 'lon': 5.61083, 'id': 1022001001},
        'Meuse at Chooz': {'lat': 50.09167, 'lon': 4.78583, 'id':1720000001},
        'Meuse at St Pieter': {'lat': 50.85000 , 'lon': 5.69417, 'id': 16},
        'Meuse at St-Mihiel': {'lat': 48.86667, 'lon': 5.52750, 'id': 12220010},
        'Rur at Stah': {'lat': 51.1, 'lon': 6.10250, 'id': 91000001},
        'Rur at Monschau': {'lat': 50.55000 , 'lon': 6.25250, 'id': 15300002},
}
#%%# We select a point for now
for loc_title in locs.keys():
    print(loc_title)
    # lat_sel = 50.8934 #50.5908#50.467
    # lon_sel = 5.728 # 5.656 #5.570
    lat_sel = locs[loc_title]['lat'] #50.8934 #50.5908#50.467
    lon_sel = locs[loc_title]['lon'] #5.728 # 5.656 #5.570

    summer_sel = summer['Q'].sel(lat=lat_sel, lon= lon_sel, method='nearest').to_dataframe()
    winter_sel = winter['Q'].sel(lat=lat_sel, lon= lon_sel, method='nearest').to_dataframe()

    #
    ams_ = pd.concat([summer_sel.rename(columns={'Q':'Q_summer'}), winter_sel.rename(columns={'Q':'Q_winter'})], axis =1 )
    ams = ams_[['Q_summer', 'Q_winter']].max(axis = 1)
    #% We do the EVA analysis
    emp_summer = calculate_emp(summer_sel.Q.values)
    emp_winter = calculate_emp(winter_sel.Q.values)
    emp_ams = calculate_emp(ams.values)

    #
    Ts = np.array(np.arange(2,2001,1))
    summ_summer, ret_levels_summer = fit_gumb_gev(summer_sel.Q.values, Ts)
    summ_winter, ret_levels_winter = fit_gumb_gev(winter_sel.Q.values, Ts)
    ams_params, ret_levels_ams = fit_gumb_gev(ams.values, Ts)

    #
    #quantiles = np.arange(rounddown(ret_levels_summer.loc[2, 'gumbel_r']),roundup(max(ret_levels_summer.loc[500, 'genextreme'], ret_levels_winter.loc[500, 'genextreme'])),5)
    quantiles = np.arange(rounddown(ret_levels_summer.loc[2, 'gumbel_r']),roundup(max(ret_levels_summer.loc[500, 'gumbel_r'], ret_levels_winter.loc[100, 'gumbel_r'], emp_winter.value.max(), emp_summer.value.max())),5)
    df_q_gumb = combine_probs_gumbel(quantiles, summ_winter, summ_summer, distr='gumbel_r')
    df_q_gev = combine_probs_gev(quantiles, summ_winter, summ_summer, distr='genextreme')

    ymin = rounddown(max(emp_winter.value.min(), emp_summer.value.min()))
    ymax = roundup(max(emp_winter.value.max(), emp_summer.value.max()))

    #
    #Plotting
    fig, ax = plt.subplots(1,1, figsize=(8,6))
    ax.plot(emp_summer.emp_rp, emp_summer.value, 'or', label='summer')
    ax.plot(emp_winter.emp_rp, emp_winter.value, 'ob', label='winter')
    ax.plot(emp_ams.emp_rp, emp_ams.value, '.k', label = "annual maxima")

    ax.plot(ret_levels_ams.index, ret_levels_ams['genextreme'], '-k', label='GEV AMs')

    #ax.plot(ret_levels_summer.index, ret_levels_summer['gumbel_r'], '-r', label='Gumbel')
    ax.plot(ret_levels_summer.index, ret_levels_summer['genextreme'], '--r', label='GEV summer')

    #ax.plot(ret_levels_winter.index, ret_levels_winter['gumbel_r'], '-b', label='Gumbel')
    ax.plot(ret_levels_winter.index, ret_levels_winter['genextreme'], '--b', label='GEV winter')

    #ax.plot(1/df_q_gumb['exc_prob'].values, df_q_gumb.index, '-g')
    ax.plot(1/df_q_gev['exc_prob'].values, df_q_gev.index, '--m', lw=2.5, label = "GEV - combined")

    plt.legend()
    plt.xscale('log')
    plt.ylabel('Discharge (m3/s)')
    plt.xlabel('Return Period (yrs)')
    plt.xlim([2,2000])
    plt.ylim([ymin, ymax])
    plt.title(loc_title)
    plt.show()

    id = locs[loc_title]['id']
    plt.savefig(os.path.join(fn_fig, f'{id}_{loc_title}_combinedGEV_analysis.png'), dpi=400)

# %%
# stacked = ds.stack(z=('time','runs')).reset_index('z')
# stacked = stacked.assign_coords({'z':np.arange(0,len(stacked['z']))})
# stacked = stacked.rename({'time':'old_time'})
# stacked = stacked.rename_dims({'z':'time'})