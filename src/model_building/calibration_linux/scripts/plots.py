""" Utils functions for plotting results to be called from the different analyze scripts """
import os
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt

import hydromt
from hydromt.stats import skills

__all__ = ["plot_model_results"]

def plot_model_results(
    obs, 
    runs, 
    station, 
    discharge_name='Q', 
    savedir=None, 
):
    """ 
    Plot observation and simulation run(s) results, including performance, for one station.
    Returns a dataframe with performance criteria (NSE, KGE, KGE_np_flood, NSElog).
    Save plots in savedir if provided.

    Parameters
    ----------
    obs: xarray.DataArray
        Observation timeseries.
    runs: dict
        Dictionnary containing different model runs results.
        Contains the following keys: 'longname' name of run, 'results' simulation timeseries (xarray.DataArray)
    station: int
        Station index at which to plot the results.
    discharge_name : str
        Name of discharge variable in run['results']. By default 'Q'.
    savedir: str, optional
        If provided, will save plots in the savedir path.
    
    Returns
    -------
    performance: pd.DataFrame
        Performance criteria for each run. Columns are: Run_name, NSE, KGE_2009, KGE_np_flood, NSE_log.

    """
    st = station
    # Read observation at the station
    obs_i = obs.sel(index=st)
    mask = ~obs_i.isnull()
    try:
        obs_i_nan = obs_i.where(mask, drop = True)
    except:
        obs_i_nan = obs_i[discharge_name].where(mask[discharge_name], drop = True)
    obs_i_nan_nm7q = obs_i_nan.rolling(time = 7).mean().resample(time = 'A').min('time')
    
    # Select a year with no data for zoom in plot
    obs_i_nanyear = obs_i.isnull().resample(time='Y').sum()
    try:
        years = obs_i_nanyear["time"][obs_i_nanyear ==0].values
    except:
        years = obs_i_nanyear["time"][obs_i_nanyear[discharge_name] ==0].values
        
    if years.size == 0:
        year = None
    else:
        year = pd.to_datetime(years[-1]).year

    # Initialise plot
    # Plot 1 is all years discharge plot
    # Plot 2 is discharge plot for a specific year
    # Plot 3 is cumulative runoff for a specific year
    n=3
    fig, axes = plt.subplots(n, 1, sharex=False, figsize=(15, n * 4))
    axes = [axes] if n == 1 else axes


    # Compute performance between obs and sim for each run (ie. ksathorfrac value)
    # Initialise performance dataframe
    perf_metrics = ["Run_name", "NSE", "KGE_2009", "KGE_np_flood", "Biais", "NSE_log", "dist_nse_nselog", "dist_nse_nselog_nsenm7q"]
    performance = pd.DataFrame(
        data = np.zeros((len(runs),len(perf_metrics))),
        columns = perf_metrics,
        dtype = "float32",
    )

    nruns = len(runs)
    colors = plt.cm.viridis_r(np.linspace(0,1,nruns))

    print(f"Plotting and computing performance metrics for station {st}")
    ind = 0
    for i, r in enumerate(runs):
        run = runs[r]
        #convert df to ds
        df = run['results'][f"{discharge_name}_{st}"]
        sim_i = df.to_xarray().to_dataset().rename({f"{discharge_name}_{st}":f"{discharge_name}"})
        sim_i = sim_i[discharge_name].sel(time=slice(obs_i.time[0], obs_i.time[-1]))
        # sim_i = run['results'][discharge_name].sel(index=st).sel(time=slice(obs_i.time[0], obs_i.time[-1]))
        try:
            sim_i_nan = sim_i.where(mask, drop = True)
        except:
            sim_i_nan = sim_i.where(mask[discharge_name], drop = True)
        nse = skills.nashsutcliffe(sim_i_nan, obs_i_nan).values.round(4)
        nse_log = skills.lognashsutcliffe(sim_i_nan, obs_i_nan).values.round(4)
        kge_2009 = skills.kge(sim_i_nan, obs_i_nan)['kge'].values.round(4)
        kge_np = skills.kge_non_parametric_flood(sim_i_nan, obs_i_nan)
        kge_np_flood = kge_np['kge_np_flood'].values.round(4)
        biais = kge_np['kge_np_flood_pearson_coef'].values.round(4)
        # add measure for nm7q
        sim_i_nan_nm7q = sim_i_nan.rolling(time = 7).mean().resample(time = 'A').min('time')
        nse_nm7q = skills.nashsutcliffe(sim_i_nan_nm7q, obs_i_nan_nm7q).values.round(4)
        dist_nse_nselog = np.sqrt((1-nse)**2 + (1-nse_log)**2)
        dist_nse_nselog_nsenm7q = np.sqrt((1-nse)**2 + (1-nse_log)**2 + (1-nse_nm7q)**2)
        # Add to performance dataframe
        performance.loc[ind, :] = [run['longname'], nse, kge_2009, kge_np_flood, biais, nse_log, dist_nse_nselog, dist_nse_nselog_nsenm7q]

        labeltxt = f"{run['longname']}, KGE_np_flood: {kge_np_flood},  KGE_2009: {kge_2009}, NSE: {nse}, NSE_log: {nse_log}"
        sim_i.plot.line(ax=axes[0], x="time", label=labeltxt, color = colors[i])

        if year is not None:
            sim_i_year = sim_i.sel(time=slice(f'{year-1}-10-01', f'{year}-09-30'))
            sim_i_year.plot(ax=axes[1], x="time", label=labeltxt, color = colors[i])
            (sim_i_year*86400).cumsum().plot(ax=axes[2], x="time", label=labeltxt, color = colors[i])

        ind+=1

        # Plot obs
    obs_i.plot.line(ax=axes[0], x="time", label='obs', color='black', linestyle = "--")
    if year is not None:
        obs_i_year = obs_i.sel(time=slice(f'{year-1}-10-01', f'{year}-09-30'))
        obs_i_year.plot(ax=axes[1], x="time", label='obs', color='black', linestyle = "--")
        (obs_i_year*86400).cumsum().plot(ax=axes[2], x="time", label='obs', color='black', linestyle = "--")


    # Plot titles
    axes[0].set_title(f"Simulated discharge at {st}")
    axes[0].set_ylabel("Discharge [m3/s]")
    # axes[0].legend()
    axes[1].set_title(f"Simulated discharge at {st} for year {year}")
    axes[1].set_ylabel("Discharge [m3/s]")
    # axes[1].legend()
    axes[2].set_title(f"Cumulative runoff at {st} for year {year}")
    axes[2].set_ylabel("Cumulative runoff [m3]")
    # axes[2].legend() #remove because too many sets 

    plt.tight_layout()

    # Save plot
    print(f"Saving results for station {st}")
    if savedir is not None:
        plt.savefig(os.path.join(savedir, f'discharge_plot_{st}.png'), dpi=150, bbox_inches='tight')

    return performance


def plot_additional_sign(
    obs, 
    runs, 
    station, 
    discharge_name='Q', 
    savedir=None, 
    performance=None,
    ):
    """ 
    Plot observation and simulation run(s) results for specific signatures for one station.
    Save plots in savedir if provided.

    Parameters
    ----------
    obs: xarray.DataArray
        Observation timeseries.
    runs: dict
        Dictionnary containing different model runs results.
        Contains the following keys: 'longname' name of run, 'results' simulation timeseries (xarray.DataArray)
    station: int
        Station index at which to plot the results.
    discharge_name : str
        Name of discharge variable in run['results']. By default 'Q'.
    savedir: str, optional
        If provided, will save plots in the savedir path.
    
    Returns
    -------
    figure
    """

    st = station
    # Read observation at the station
    obs_i = obs.sel(index=st)
    mask = ~obs_i.isnull()
    try:
        obs_i_nan = obs_i.where(mask, drop = True)
    except:
        obs_i_nan = obs_i[discharge_name].where(mask[discharge_name], drop = True)
    #obs max
    obs_i_nan_max = obs_i_nan.sel(time = slice(f"{str(obs_i_nan['time.year'][0].values)}-09-01", f"{str(obs_i_nan['time.year'][-1].values)}-08-31")).resample(time = 'AS-Sep').max('time')
    #nm7q
    obs_i_nan_nm7q = obs_i_nan.rolling(time = 7).mean().resample(time = 'A').min('time')

    # Initialise plot
    # Plot 1 is discharge regime
    # Plot 2 is NSE versus NSElog for ksathorfrac values
    # Plot 3 is plotting position high flows 
    # plot 4 is plotting position low flows

    fig, axes = plt.subplots(2, 2, sharex=False, figsize=(16/2.54, 16/2.54))
    ax = axes.flatten()

    # Compute performance between obs and sim for each run (ie. ksathorfrac value)

    nruns = len(runs)
    colors = plt.cm.viridis_r(np.linspace(0,1,nruns))
    fs = 8

    #gumbel params
    a=0.3
    b = 1.-2.*a

    print(f"Plotting and computing performance metrics for station {st}")
    for i, r in enumerate(runs):
        run = runs[r]
        #convert df to ds
        df = run['results'][f"{discharge_name}_{st}"]
        sim_i = df.to_xarray().to_dataset().rename({f"{discharge_name}_{st}":f"{discharge_name}"})
        sim_i = sim_i[discharge_name].sel(time=slice(obs_i.time[0], obs_i.time[-1]))
        try:
            sim_i_nan = sim_i.where(mask, drop = True)
        except:
            sim_i_nan = sim_i.where(mask[discharge_name], drop = True)

        #max
        sim_i_nan_max = sim_i_nan.sel(time = slice(f"{str(sim_i_nan['time.year'][0].values)}-09-01", f"{str(sim_i_nan['time.year'][-1].values)}-08-31")).resample(time = 'AS-Sep').max('time')   
        #nm7q
        sim_i_nan_nm7q = sim_i_nan.rolling(time = 7).mean().resample(time = 'A').min('time')

        nse = skills.nashsutcliffe(sim_i_nan, obs_i_nan).values.round(4)
        nse_log = skills.lognashsutcliffe(sim_i_nan, obs_i_nan).values.round(4)
        kge_2009 = skills.kge(sim_i_nan, obs_i_nan)['kge'].values.round(4)
        kge_np = skills.kge_non_parametric_flood(sim_i_nan, obs_i_nan)
        kge_np_flood = kge_np['kge_np_flood'].values.round(4)
        biais = kge_np['kge_np_flood_pearson_coef'].values.round(4)

        if performance is not None:
            dist_nse_nselog_min = performance["dist_nse_nselog_nsenm7q"].min()
            ksathorfrac_dist_min = performance["dist_nse_nselog_nsenm7q"].idxmin()
            
        # labeltxt = f"{run['longname']}, NSE: {nse}, NSE_log: {nse_log}"
        labeltxt = f"{run['longname']}"

        # ax1 dis regime
        sim_i_nan.groupby("time.month").mean("time").plot.line(ax=ax[0], x="month", label=labeltxt, color = colors[i])
        #plot best k in red
        if performance is not None:
            if run['longname'] == f"KsH{ksathorfrac_dist_min}":
                sim_i_nan.groupby("time.month").mean("time").plot.line(ax=ax[0], x="month", label=labeltxt, color = "r")
        
        # ax2 nse versus nse log
        ax[1].plot(nse, nse_log, marker = ".", linestyle = "None", label = labeltxt, color = colors[i])


        #gumbel high ax2  
        max_y = np.round(max(obs_i_nan.max().values, sim_i_nan.max().values))
        ymin, ymax = 0, max_y
        p1 = ((np.arange(1,len(sim_i_nan_max.time)+1.)-a))/(len(sim_i_nan_max.time)+b)
        RP1 = 1/(1-p1)
        gumbel_p1 = -np.log(-np.log(1.-1./RP1))
        ts = [2., 5.,10.,30.] #,30.,100.,300.,1000.,3000.,10000.,30000.]
        #plot
        ax[2].plot(gumbel_p1, sim_i_nan_max.sortby(sim_i_nan_max), marker = '.', color = colors[i], linestyle = 'None', label = labeltxt, markersize = 4)
        #plot best k in red
        if performance is not None:
            if run['longname'] == f"KsH{ksathorfrac_dist_min}":
                ax[2].plot(gumbel_p1, sim_i_nan_max.sortby(sim_i_nan_max), marker = '.', color = "r", linestyle = 'None', label = labeltxt, markersize = 4)



        #gumbel low ax3
        max_ylow = np.round(max(obs_i_nan_nm7q.max().values, sim_i_nan_nm7q.max().values))
        ymin, ymax = 0, max_ylow
        p1 = ((np.arange(1,len(sim_i_nan_nm7q.time)+1.)-a))/(len(sim_i_nan_nm7q.time)+b)
        RP1 = 1/(1-p1)
        gumbel_p1_low = -np.log(-np.log(1.-1./RP1))
        ts = [2., 5.,10.,30.] #,30.,100.,300.,1000.,3000.,10000.,30000.]
        #plot
        ax[3].plot(gumbel_p1_low, sim_i_nan_nm7q.sortby(sim_i_nan_nm7q, ascending=False), marker = '.', color = colors[i], linestyle = 'None', label = labeltxt, markersize = 4)
        #plot best k in red
        if performance is not None:
            if run['longname'] == f"KsH{ksathorfrac_dist_min}":
                ax[3].plot(gumbel_p1_low, sim_i_nan_nm7q.sortby(sim_i_nan_nm7q, ascending=False), marker = '.', color = "r", linestyle = 'None', label = labeltxt, markersize = 4)
        

    # Plot obs
    #dis regime
    obs_i_nan.groupby("time.month").mean("time").plot.line(ax=ax[0], x="month", label='obs', color='black', linestyle = "--")
    # ax1 is nse versus nse_log
    # ax2 is gumbel high
    ax[2].plot(gumbel_p1, obs_i_nan_max.sortby(obs_i_nan_max), marker = '+', color = 'k', linestyle = 'None', label = 'Obs.', markersize = 6)
    # ax3 is gumbel low
    ax[3].plot(gumbel_p1_low, obs_i_nan_nm7q.sortby(obs_i_nan_nm7q, ascending=False), marker = '+', color = 'k', linestyle = 'None', label = 'Obs.', markersize = 6)


    for t in ts:
        ax[2].vlines(-np.log(-np.log(1-1./t)),ymin,max_y,'0.5', alpha=0.4)
        ax[2].text(-np.log(-np.log(1-1./t)),max_y*0.2,'T=%.0f y' %t, rotation=45, fontsize = fs)

        ax[3].vlines(-np.log(-np.log(1-1./t)),ymin,max_ylow,'0.5', alpha=0.4)
        ax[3].text(-np.log(-np.log(1-1./t)),max_ylow*0.8,'T=%.0f y' %t, rotation=45, fontsize = fs)

        
   # Plot titles and labels
    ax[0].set_title(f"Simulated streamflow at {st}", fontsize = fs)
    ax[0].set_ylabel("Streamflow [m3/s]", fontsize = fs)
    ax[0].set_xlabel("", fontsize = fs)

    if performance is not None:
        ax[1].set_title(f"Best ksathorfrac = {ksathorfrac_dist_min}", fontsize = fs)
    ax[1].set_xlabel('NSE', fontsize = fs)
    ax[1].set_ylabel('NSE log', fontsize = fs)

    ax[2].set_ylabel('max. annual Q (m$^3$s$^{-1}$)', fontsize = fs)
    ax[2].set_xlabel('Plotting position and associated return period', fontsize = fs)
   

    ax[3].set_ylabel('NM7Q (m$^3$s$^{-1}$)', fontsize = fs)
    ax[3].set_xlabel('Plotting position and associated return period', fontsize = fs)     

    # ax[1].legend(fontsize = fs)  #remove too many sets
    ax[1].set_xlim([-0.5,1])
    ax[1].set_ylim([-0.5,1])
    # ax[2].legend()

    for a in ax:
        a.tick_params(axis = "both", labelsize = fs)

    plt.tight_layout()

    # Save plot
    print(f"Saving results for station {st}")
    if savedir is not None:
        plt.savefig(os.path.join(savedir, f'signature_plot_{st}.png'), dpi=150, bbox_inches='tight')