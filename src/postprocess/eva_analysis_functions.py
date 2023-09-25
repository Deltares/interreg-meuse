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


# %% Defining functions
def plot_return_period_tradi(ax, cmap, norm, data_Tcal, ds_stats, ylims, xlims, station, distr, title):
        bm_summary = data_Tcal[station]['MOD']['BM_fit']
        #Extract the value of the shape parameter
        param = list()
        # for col in cols:
        #         param.loc[int(col), param_type] = bm_summary[col]['eva_params'].loc[param_type,distr]

        #Plotting results - return levels per 65 years ensemble
        for n in np.arange(1, 17, 1):
                #print(n)
                ax.plot(bm_summary[f"65_r{n}"]['return_levels'].index, bm_summary[f"65_r{n}"]['return_levels'][distr], '-k', zorder = 2)
                ax.fill_between(bm_summary[f"65_r{n}"]['return_levels'].index.values, bm_summary[f"65_r{n}"]['conf_int_low'][distr].values.astype(float), bm_summary[f"65_r{n}"]['conf_int_high'][distr].values.astype(float), color='grey', edgecolor=None, alpha=0.1, zorder = 1)
                param_value = bm_summary[f"65_r{n}"]['eva_params'].loc["shape",distr]
                param.append(param_value)

        if isinstance(ds_stats, xr.Dataset): #Official statistics
                if station == 91000001:
                        station_stats = 24
                else:
                        station_stats = station
                if station_stats in ds_stats.coords['stations'].data:
                        df_stats = ds_stats.sel(stations=station_stats).to_dataframe()
                        ax.plot(df_stats.index, df_stats['valeur'].values, 'or', zorder = 5, label = 'reported stat.')
                        ax.vlines(x= df_stats.index, ymin = df_stats['int_conf_bas'].values, ymax = df_stats['int_conf_haut'].values, color='r', zorder = 5)                
        #Observed time series
        ax.scatter(data_Tcal[station]["OBS"]['emp_T'].emp_rp.values, data_Tcal[station]["OBS"]['emp_T'].value, c = data_Tcal[station]["OBS"]['emp_month'].values, marker = 'X', label='observed', edgecolors= 'k', linewidth=0.3, cmap = cmap, norm=norm, zorder = 7)
        ax.hlines(y=data_Tcal[station]["OBS"]['max_2021'], xmin=xlims[0], xmax=xlims[1], color = 'chartreuse', label = '2021 level')

        m = ax.scatter(data_Tcal[station]["MOD"]['emp_T'].emp_rp.values, data_Tcal[station]["MOD"]['emp_T'].value, c = data_Tcal[station]["MOD"]['emp_month'].values, marker = 'o', label='modelled', edgecolors= 'k', linewidth=0.3, cmap = cmap, norm=norm, zorder = 6)        
        ax.plot(bm_summary[str(1024)]['return_levels'].index, bm_summary[str(1024)]['return_levels'][distr], '-r', label=distr, zorder = 4)
        ax.fill_between(bm_summary[str(1024)]['return_levels'].index.values, bm_summary[str(1024)]['conf_int_low'][distr].values.astype(float), bm_summary[str(1024)]['conf_int_high'][distr].values.astype(float), color='darkred', edgecolor=None, alpha=0.5, zorder = 3)        
        ax.set_xscale('log')
        ax.set_ylim(ylims)
        ax.set_xlim(xlims)
        ax.set_xlabel('Return Period (year)')
        ax.set_ylabel('Discharge ($m^3/s$)')
        if distr == 'genextreme':
                ax.text(0.70,0.05, f'$\gamma$: [{min(param):.2f} - {max(param):.2f}]', fontdict={'fontsize':'small'},transform=ax.transAxes)
        #ax.legend()
        ax.set_title(title)


def plot_gumbel_gev_n(ax, cmap, norm, data_Tcal, ds_stats, n, ylims, xlims, station, title):
        bm_summary = data_Tcal[station]['MOD']['BM_fit']  
        param = bm_summary[str(n)]['eva_params'].loc["shape",'genextreme']      
        ax.plot(bm_summary[str(n)]['return_levels'].index, bm_summary[str(n)]['return_levels']['gumbel_r'], '-',color = 'C1', label='Gumbel')
        ax.plot(bm_summary[str(n)]['return_levels'].index, bm_summary[str(n)]['return_levels']['genextreme'], '-g', label='GEV')
        ax.fill_between(bm_summary[str(n)]['return_levels'].index.values, bm_summary[str(n)]['conf_int_low']['gumbel_r'].values.astype(float), bm_summary[str(n)]['conf_int_high']['gumbel_r'].values.astype(float), color='C1', edgecolor=None, alpha=0.1)
        ax.fill_between(bm_summary[str(n)]['return_levels'].index.values, bm_summary[str(n)]['conf_int_low']['genextreme'].values.astype(float), bm_summary[str(n)]['conf_int_high']['genextreme'].values.astype(float), color='green', edgecolor=None, alpha=0.1)
        m = ax.scatter(data_Tcal[station]["MOD"]['emp_T'].emp_rp.values, data_Tcal[station]["MOD"]['emp_T'].value, c = data_Tcal[station]["MOD"]['emp_month'].values, marker = 'o', label='modelled', edgecolors= 'k', linewidth=0.3, cmap = cmap, norm=norm, zorder = 6)        
        if isinstance(ds_stats, xr.Dataset): #Official statistics
                if station == 91000001:
                        station_stats = 24
                else:
                        station_stats = station
                if station_stats in ds_stats.coords['stations'].data:
                        df_stats = ds_stats.sel(stations=station_stats).to_dataframe()
                        ax.plot(df_stats.index, df_stats['valeur'].values, 'or', zorder = 5, label = 'reported stat.')
                        ax.vlines(x= df_stats.index, ymin = df_stats['int_conf_bas'].values, ymax = df_stats['int_conf_haut'].values, color='r', zorder = 5)          
        #Observed time series
        ax.scatter(data_Tcal[station]["OBS"]['emp_T'].emp_rp.values, data_Tcal[station]["OBS"]['emp_T'].value, c = data_Tcal[station]["OBS"]['emp_month'].values, marker = 'X', label='observed', edgecolors= 'k', linewidth=0.3, cmap = cmap, norm=norm, zorder = 7)
        ax.hlines(y=data_Tcal[station]["OBS"]['max_2021'], xmin=xlims[0], xmax=xlims[1], color = 'chartreuse', label = '2021 level')

        ax.set_xscale('log')
        ax.set_ylim(ylims)
        ax.set_xlim(xlims)
        ax.set_xlabel('Return Period (year)')
        ax.set_ylabel('Discharge ($m^3/s$)')
        ax.text(0.82,0.05, f'$\gamma$ = {param:.2f}', fontdict={'fontsize':'small'},transform=ax.transAxes)
        # produce a legend with the unique colors from the scatter
        #legend1 = ax.legend(*m.legend_elements(), title="Month", loc="upper left", fontsize = 7)
        #ax.add_artist(legend1)
        #ax.legend()
        ax.set_title(title)


def plot_convergence_params(ax, data_Tcal, station, T_considered, cols, distr, title):
        color=iter(cm.rainbow(np.linspace(0,1,len(T_considered)+1)))
        bm_summary = data_Tcal[station]['MOD']['BM_fit']

        for T_i in T_considered:
                c=next(color)
                unc = pd.DataFrame(columns = cols, index= ['ret_value', 'cil', 'cih'])
                for key in cols:
                        unc.loc['ret_value',key]=  bm_summary[key]['return_levels'].loc[T_i, distr]
                        unc.loc['cil',key]=  bm_summary[key]['conf_int_low'].loc[T_i, distr]
                        unc.loc['cih',key]=  bm_summary[key]['conf_int_high'].loc[T_i, distr]
                ax.plot([int(i) for i in unc.columns],unc.loc['ret_value',:],'o-', color=c, label = str(T_i))
                ax.fill_between([int(i) for i in unc.columns],unc.loc['cil',:].values.astype(float),unc.loc['cih',:].values.astype(float),color=c, alpha=0.15, edgecolor=None)                
                # if isinstance(obs_t[station], pd.DataFrame):
                #         try:
                #                 ax.plot([int(i) for i in unc.columns], np.repeat(obs_t[station]['valeur'].loc[float(T_i)], len([int(i) for i in unc.columns])), '--', color=c)
                #         except:
                #                 continue
        ax.legend()
        ax.set_xlabel("Record length (years)")
        ax.set_ylabel('Discharge ($m^3/s$)')
        ax.set_title(title) #locs[station]+' - '+ distr

def plot_convergence_params_type(ax, data_Tcal, station, cols, distr, param_type, title):
        bm_summary = data_Tcal[station]['MOD']['BM_fit']
        len_record = [int(i) for i in cols]
        param = pd.DataFrame(index = len_record, columns=[param_type])
        for col in cols:
                param.loc[int(col), param_type] = bm_summary[col]['eva_params'].loc[param_type,distr]
        ax.plot(param.index,param,'o-', color='k')
        ax.set_xlabel("Record length (years)")
        ax.set_ylabel(f'{param_type} value')
        ax.set_title(title) #locs[station]+' - '+ distr


def roundup(x):
        return int(math.ceil(x / 100.0)) * 100

def calculate_emp(data):
    emp_p = pd.DataFrame(data=data, columns=['value'])
    emp_p['rank'] = emp_p.iloc[:,0].rank(axis=0, ascending=False, method = 'dense')
    emp_p['exc_prob'] = (emp_p['rank']-0.3)/(emp_p['rank'].size+0.4) #change this line with what Anaïs sends to me, but is already correct
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
# %%
