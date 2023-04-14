# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 11:44:32 2022

@author: bouaziz
"""
#%%
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import hydromt 
import glob
from hydromt_wflow import WflowModel

fs = 8

Folder_p = r"p:\11208719-interreg\wflow\d_manualcalib"

Folder_p_fews_data= r"p:\archivedprojects\11205237-grade\wflow\wflow_meuse_julia\data_juli_2021\fews_export"
Folder_p_fews_data_jan= r"p:\archivedprojects\11205237-grade\wflow\wflow_meuse_julia\data_juli_2021\vanJan"
Folder_p_spw_download= r"p:\archivedprojects\11205237-grade\wflow\wflow_meuse_julia\data_juli_2021\SPW_website_download"

Folder_plots = r"d:\interreg\Plots\july_2021_routing"


#%% model

runs = {

    "kinematic": {
        "longname": "kinematic",
        "color": "#a6cee3",
        "case": "run_manualcalib_july_2021_kinematic",
        "config_fn": "run_manualcalib_july_2021_regnie.toml",
            },

    "loc.iner": {
        "longname": "loc.iner",
        "color": "#1f78b4",
        "case": "run_manualcalib_july_2021_1d",
        "config_fn": "run_manualcalib_july_2021_regnie.toml",
             },
    
    "loc.iner.flpl1d": {
        "longname": "loc.iner.flpl1d",
        "color": "#b2df8a",
        "case": "run_manualcalib_july_2021",
        "config_fn": "run_manualcalib_july_2021_regnie.toml",
             },
    
    "loc.iner1d2d": {
        "longname": "loc.iner1d2d",
        "color": "#33a02c",
        "case": "run_manualcalib_july_2021_1d2d",
        "config_fn": "run_manualcalib_july_2021_regnie.toml",
             },
    
}

for r in runs:
    print(r)
    run = runs[r]
    case = run["case"]
    
    root = os.path.join(Folder_p, case)
    mod = WflowModel(root=root, mode="r", config_fn=run["config_fn"])

    runs[r].update({"mod": mod})
    res = runs[r]["mod"].results
    runs[r].update({"res": res})

#%%
#read csv directly to later plot precip (ids are then P_{id} for full upstream precip)
df_kin = pd.read_csv(os.path.join(Folder_p, "run_manualcalib_july_2021_kinematic", "run_july2021_regnie", "output.csv"), index_col=0, parse_dates=True, header=0)
#todo: also add the output file from the radar run to have precipitation averaged over each subcatchment from this dataframe
# df_kin_radar = pd.read_csv(os.path.join(Folder_p, "run_july2021_radar", "output.csv"), index_col=0, parse_dates=True, header=0)


stations_dic = {
    "Meuse at Goncourt" : 1011,
    "Mouzon at Circourt-sur-Mouzon" : 1013, 
    "Vair at Soulosse-sous-Saint-Elophe" : 1016, 
    "Meuse at Saint-Mihiel" : 101, 
    "Meuse at Stenay" : 3,
    "Bar at Cheveuges" : 41, 
    "Vence at Francheville" : 42, 
    "Sormonne at Belval" : 43, 
    "Semois at Membre Pont" : 5, 
    "Semois at Sainte-Marie" : 503, 
    "Vierre at Straimont" : 501,
    "Chiers at Carignan" : 201, 
    "Chiers at Longlaville" : 203, 
    "Crusnes at Pierrepont" : 206, 
    "Ton at Écouviez" : 207, 
    "Loison at Han-les-Juvigny" : 209,
    "Viroin at Treignes" : 6,
    "Meuse at Chooz" : 4,
    "Lesse at Daverdisse" : 802, 
    "Lhomme at Jemelle" : 803, 
    "Lesse at Gendron" : 801,
    "Hermeton at Hastiere" : 701, 
    "Bocq at Yvoir" : 702, 
    "Molignee at Warnant" : 703, 
    "Hoyoux at Modave" : 704, 
    "Ourthe Occidentale at Ortho" : 1002, 
    "Ourthe Orientale at Mabompre" : 1003, 
    "Ourthe at Tabreux" : 10, 
    "Hantes at Wiheries" : 903, 
    "Sambre at Salzinnes" : 9,
    "Mehaigne at Huccorgne" : 13,
    "Meuse at Amay" : 1401,
    "Ambleve at Martinrive" : 11, 
    "Vesdre at Chaudfontaine" : 12,
    "Meuse at Borgharen" : 16}

stations_dic_rev = dict((v, k) for k, v in stations_dic.items())


#%% observed data: available from 2 sources: fews and from a download of SPW and HYDRO BANQUE and RWSINFO.

#fews system albrecht (deltares database after event) - last year analysis
# observations = pd.read_excel(r"d:\GRADE\Data\Meuse_July2021.xlsx", skiprows = [0,1,3], parse_dates=True, index_col = 0)

#Fews system indra (RWS database)
# df = pd.read_csv(r"d:\GRADE\Data\data_fews_2021\Borgharen-Dorp.csv", index_col=0, parse_dates=True, header=0, skiprows=1)
df_obs_fews = pd.read_csv(os.path.join(Folder_p_fews_data, "Qobs_fews.csv"), index_col=0, parse_dates=True, header=0, skiprows=[1])
#clean up the file and remove unused stations
cols = df_obs_fews.columns
cols_1 = cols[~cols.str.contains("quality")]
df_obs_fews = df_obs_fews[cols_1]

#mapping dictionary
stations_obs = {
    
    10 : "H-MS-0020", #'Tabreux',
    4 :  "H-MS-0011", #'Chooz (Ile Graviat)',
    12 : "H-MS-0010", #'Chaudfontaine',
    11 : "H-MS-0017", #'Martinrive',
    5 : "H-MS-0018", #'Membre pont',
    801 : "H-MS-0013", #'Gendron',
    9 : "H-MS-0019", #'Salzinnes',
    1401: 'H-MS-0008', #amay
    43: 'Belval', 
    201: 'H-MS-CARI', 
    41: 'Cheveuges', 
    1013: 'Circourt_Mouzon', 
    207: 'Ecouviez', 
    1011: 'Goncourt', 
    209: 'Han_les_Juvigny', 
    42: 'Lafrancheville',
    203: 'Longlaville', 
    101: 'H-MS-STMI', 
    3: 'H-MS-STEN', 
    16: 'H-MS-SINT', #sint pieter vgl me t borgharen dorp -- 16 voor St Pieter
    
    #extra niet in wflow
    1: 'H-MS-0014', #Haccourt
    0: "H-MS-BORD", # 'Borgharen-Dorp' krijgt 0 -- meting was uitgevallen...
    
    }

df_obs_fews = df_obs_fews.rename(columns=dict((v, k) for k, v in stations_obs.items()))
df_obs_fews = df_obs_fews.replace(-999., np.nan)
df_obs_fews = df_obs_fews.astype(float)

#check
# df_obs_fews.plot()

#%%
#downloads SPW website:
#staan allemaal in Folder_p_spw_download, nog geen datatset - maar wordt in loop van plaatjes ingelezen. 
df = pd.read_csv(os.path.join(Folder_p_spw_download, "Q", "SPW_Q_CHAUDFONTAINE Pisc_62281002.csv"), index_col=0, parse_dates=True, header=0)
# df = pd.read_csv(r"p:\11205237-grade\wflow\wflow_meuse_julia\data_juli_2021\SPW_website_download\Q\SPW_Q_TABREUX_59211002.csv", index_col=0, parse_dates=True, header=0)

#downalod french website 2005-2022
qobs_h_fr = xr.open_dataset(r"p:\11208719-interreg\data\observed_streamflow_grade\FR-Hydro-hourly-2005_2022.nc")

#%% load catchment areas to get Q in mm/h

# Q in mm
geoms = runs["kinematic"]["mod"].staticgeoms
gdf_gauges = geoms["gauges_Sall"]
gdf_gauges = gdf_gauges.set_index("wflow_id")

xs, ys = np.vectorize(lambda p: (p.xy[0][0], p.xy[1][0]))(gdf_gauges["geometry"])
idxs_gauges = mod.staticmaps.raster.xy_to_idx(xs, ys)

mod = runs["kinematic"]["mod"]
uparea=(["index"], mod.staticmaps['wflow_uparea'].values.flat[idxs_gauges])
index = gdf_gauges.index.values.astype('int32')
df_area = pd.DataFrame(index = index, data = (mod.staticmaps['wflow_uparea'].values.flat[idxs_gauges])*1e6, columns = ["area"])


# runs["sbm_kin"]["mod"].staticmaps


#%% plots

start = "2021-07-13 06:00"
stop = "2021-07-19 06:00" 

runs_sel = ["kinematic", "loc.iner", "loc.iner.flpl1d", "loc.iner1d2d"]

for wflow_id in list(stations_obs.keys())[:-2]:
    station_name = stations_dic_rev[wflow_id]
    # print (station_name)
    print(wflow_id)

    #table with peak magn, timing and 
    df = pd.DataFrame(index = runs_sel + ["HBV", "Obs", "Obs_download"], columns = ["RC", "peak_magn", "peak_timing"])

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(16/2.54, 16/2.54))
    # ax = axes.flatten()
    #plot hydro
    for r in runs_sel:
        #hydro
        runs[r]["res"]["Q_gauges_Sall"].sel(index=wflow_id, time=slice(start, stop)).plot(ax=axes[1], color= runs[r]['color'], label = runs[r]['longname'], linewidth = 0.8)    
        #max info 
        ds_run = runs[r]["res"]["Q_gauges_Sall"].sel(index=wflow_id, time=slice(start, stop))
        max_run = ds_run.isel(time = ds_run.sel(time=slice(start,stop)).argmax().values)
        df.loc[r, "peak_magn"] = np.round(max_run.values,)
        df.loc[r, "peak_timing"] = max_run.time.values    
    
    #plot obs
    ds_obs_fews = df_obs_fews[wflow_id].to_xarray()
    ds_obs_fews.name = str(wflow_id)
    ds_obs_fews.sel(index=slice(start,stop)).plot(ax=axes[1], color = "k", label = "Obs", linewidth = 0.8, linestyle = "--")
    ds_obs_fews_mmh = ds_obs_fews / df_area.loc[wflow_id].values * 3600 * 1000
    #max 
    if ds_obs_fews.sel(index=slice(start,stop)).max("index") > 0:
        max_obs = ds_obs_fews.sel(index=slice(start,stop)).isel(index = ds_obs_fews.sel(index=slice(start,stop)).argmax().values)
        df.loc["Obs", "peak_magn"] = np.round(max_obs.values,)
        df.loc["Obs", "peak_timing"] = max_obs.index.values
    
    #also plot obs from website download - french stations
    if wflow_id in qobs_h_fr.wflow_id:
        qobs_h_fr["Q"].sel(wflow_id=wflow_id, time = slice(start,stop)).plot(ax=axes[1], color = "darkgrey", label = "Obs_download", linewidth = 0.8, linestyle = "--")
        #max
        max_fr = qobs_h_fr["Q"].sel(wflow_id=wflow_id, time = slice(start,stop)).isel(time =  qobs_h_fr["Q"].sel(wflow_id=wflow_id, time = slice(start,stop)).argmax().values)
        df.loc["Obs_download", "peak_magn"] = np.round(max_fr.values,)
        df.loc["Obs_download", "peak_timing"] = max_fr.time.values

    #also plot obs from website download - belgian stations
    filenames= glob.glob(os.path.join(Folder_p_spw_download, "Q", f"*{station_name.split(' at ')[-1]}*.csv"))
    if len(filenames)>0:
        df_spw = pd.read_csv(filenames[0], index_col=0, parse_dates=True, header=0)
        df_spw["Q"].to_xarray().sel(index = slice(start,stop)).plot(ax=axes[1], color = "darkgrey", label = "Obs_download", linewidth = 0.8, linestyle = "--")
        ds_spw = df_spw["Q"].to_xarray().sel(index = slice(start,stop))
        ds_spw = ds_spw.rename({"index": "time"})
        ds_spw_mmh = ds_spw / df_area.loc[wflow_id].values * 3600 * 1000
        #max 
        max_spw = ds_spw.isel(time = ds_spw.sel(time=slice(start,stop)).argmax().values)
        df.loc["Obs_download", "peak_magn"] = np.round(max_spw.values,)
        df.loc["Obs_download", "peak_timing"] = max_spw.time.values
     
    #plot precip
    #index not the same problem for sharex = true - convert to dataaray
    # regnie data
    ds_kin_precip = df_kin[f"P_{wflow_id}"].loc[start:stop].to_xarray()    
    ds_kin_precip.plot(ax=axes[0], color= "darkblue", label = "precip regnie", linewidth = 0.8)

    # radar data
    # ds_radar_precip = df_kin_radar[f"P_{wflow_id}"].loc[start:stop].to_xarray()
    # ds_radar_precip.plot(ax=axes[0], color= "purple", label = "precip radar", linewidth = 0.8)
    
    #plot precip cumulative op tweede as
    ax2= axes[0].twinx()
    ds_kin_precip.cumsum("time").plot(ax=ax2, color= "darkblue", label = "precip regnie", linewidth = 1, linestyle = "--")
    # ds_radar_precip.cumsum("time").plot(ax=ax2, color= "purple", label = "precip radar", linewidth = 1, linestyle = "--")
    #plot cumulative Qobs from download
    if wflow_id in qobs_h_fr.wflow_id:
        qobs_h_fr_mmh = qobs_h_fr["Q"].sel(wflow_id=wflow_id, time = slice(start,stop)) / df_area.loc[wflow_id].values * 3600 * 1000
        rc_obs_fr = qobs_h_fr_mmh.sum("time") / ds_kin_precip.sum("time")
        qobs_h_fr_mmh.cumsum("time").plot(ax=ax2, color = "darkgrey", label = f"Obs_download (RC={rc_obs_fr.values:.2f})", linewidth = 0.8, linestyle = "--")
    if len(filenames)>0: #if a spw station
        rc_obs_spw = ds_spw_mmh.sel(time = slice(start,stop)).sum("time") / ds_kin_precip.sum("time")
        ds_spw_mmh.sel(time = slice(start,stop)).cumsum("time").plot(ax=ax2, color = "darkgrey", label = f"Obs_download (RC={rc_obs_spw.values:.2f})", linewidth = 0.8, linestyle = "--")
    #plot culumative Qobs from fews 
    rc_obs_fews = ds_obs_fews_mmh.sel(index=slice(start,stop)).sum("index") / ds_kin_precip.sum("time")
    ds_obs_fews_mmh.sel(index=slice(start,stop)).cumsum("index").plot(ax=ax2, color = "k", label = f"Obs (RC={rc_obs_fews.values:.2f})", linewidth = 0.8, linestyle = "--")
    df.loc["Obs", "RC"] = np.round(rc_obs_fews.values,2)
    
    #plot cumulative Qmod 
    for r in runs_sel:
        #hydro
        mod_mmh = runs[r]["res"]["Q_gauges_Sall"].sel(index=wflow_id, time=slice(start, stop)) / df_area.loc[wflow_id].values * 3600 * 1000
        rc_mod = mod_mmh.sel(time=slice(start,stop)).sum("time") / ds_kin_precip.sum("time")
        mod_mmh.sel(time=slice(start,stop)).cumsum("time").plot(ax=ax2, color= runs[r]['color'], label = runs[r]['longname'] + f" (RC={rc_mod.values:.2f})", linewidth = 0.8)    
        df.loc[r, "RC"] = np.round(rc_mod.values,2)
    
    axes[0].set_ylabel("P [mm h${-1}$]", fontsize = fs)
    ax2.set_ylabel("P and Q (cumulative)", fontsize = fs)
    axes[1].set_ylabel("Q [m${3}$ s$^{-1}$]", fontsize = fs)

    axes[0].set_xlabel("", fontsize = fs)
    axes[1].set_xlabel("", fontsize = fs)
    
    axes[1].legend(fontsize=fs)
    
    for ax in axes:
        ax.tick_params(labelsize = fs, axis="both")
    ax2.tick_params(labelsize = fs, axis="both")
    ax2.set_title("")
    ax2.legend(fontsize=fs)
    
    axes[0].set_title(f"{station_name}", fontsize = fs)
    axes[1].set_title("", fontsize = fs)

    plt.tight_layout()
    plt.savefig(os.path.join(Folder_plots, f"hydro_precip_{station_name}_{wflow_id}.png"), dpi=300)
    #%%
    #percentage diff peak timing and magntidue
    # df["peak_magn_diff"] = df["peak_magn"] / df["peak_magn"].loc[["Obs", "Obs_download"]].max()
    # df.dropna().round({'peak_magn_diff': 2})
    
    # df["peak_timing_diff"] = int((df["peak_timing"] - df["peak_timing"].loc["Obs"])/3600/1e9)
    # df.to_csv(os.path.join(Folder_plots, f"hydro_info_{station_name}_{wflow_id}.csv"))

    
# df["Q.m"].plot(ax=ax[i], linestyle = "--", color="k", label = "Obs.", linewidth = 0.8)
# df["Q"].plot(ax=ax[i], linestyle = "--", color="k", label = "Obs.", linewidth = 0.8)



#%% plot contour maps of precipitation sum

inmaps_regnie = xr.open_dataset(r"p:\11208719-interreg\data\july_2021\forcing_regnie_19072021_28032021.nc")
inmaps_radar = xr.open_dataset(r"p:\11208719-interreg\data\july_2021\radar_july_2021\b_preprocess\forcing_regnie_radar.nc")

prec_event_regnie = inmaps_regnie["precipitation"].sel(time=slice("2021-07-13 06:00:00", "2021-07-16 06:00:00")).sum("time")
prec_event_radar = inmaps_radar["precipitation"].sel(time=slice("2021-07-13 06:00:00", "2021-07-16 06:00:00")).sum("time")

# enforce same maximum and minimum for scale
vmax = max(prec_event_regnie.max(), prec_event_radar.max())
vmin = min(prec_event_regnie.min(), prec_event_radar.min())

fig, ax = plt.subplots()
cmap = plt.get_cmap("viridis", 11)
prec_event_regnie.plot(ax=ax, cmap=cmap, vmax=vmax, vmin=vmin)
geoms["subcatch_Sall"].plot(ax=ax, facecolor="None")
plt.title("P sum 13-07-21 06:00 to 16-07-21 06:00")
plt.tight_layout()
plt.savefig(os.path.join(Folder_plots, "regnie_precip_sum.png"), dpi=300)

fig, ax = plt.subplots()
prec_event_radar.plot(ax=ax, cmap=cmap)
geoms["subcatch_Sall"].plot(ax=ax, facecolor="None")
plt.title("P sum 13-07-21 06:00 to 16-07-21 06:00")
plt.tight_layout()
plt.savefig(os.path.join(Folder_plots, "radar_precip_sum.png"), dpi=300)

#%%
plt.figure(); mod.staticmaps['ksathorfrac_sub'].plot()

#%% plot contributions

r = "sbm_regnie"

#boven borgharen
df_c = pd.DataFrame()
for i, wflow_id in enumerate([ 11, 12, 10, 801, 13, 701, 703, 9,  4]): #41,42,43,101,3,201,5,6 
    station_name = stations_dic_rev[wflow_id]
    df_ = runs[r]["res"]["Q_gauges_Sall"].sel(index=wflow_id , time=slice(start, stop)).to_dataframe()["Q_gauges_Sall"]
    df_ = df_.rename(str(station_name)) #index={f"{var}_geul_stations":wflow_id})
    df_c = pd.concat([df_c,df_], axis=1)

fig, (ax1,ax2) = plt.subplots(1,2, figsize = (16/2.54, 10/2.54))    
df_c.plot.area(ax=ax1)
runs[r]["res"]["Q_gauges_Sall"].sel(index=16 , time=slice(start, stop)).to_dataframe()["Q_gauges_Sall"].plot(ax=ax1, linestyle="--", color="k", label = "Borgharen")
ax1.set_ylim([0,3500])
ax1.set_ylabel("Q (m$^{3} s^{-1}$)", fontsize = fs)
ax1.set_title("Main contributions for the Meuse at Borgharen", fontsize = fs)
# plt.savefig(os.path.join(Folder_plots, "July2021_stacked_flow_chooz.png"), dpi=300)
ax1.tick_params(axis = "both", labelsize = fs)
ax1.legend(fontsize = 5)
ax1.set_xlabel("")

#boven chooz
df_c = pd.DataFrame()
for i, wflow_id in enumerate([ 41,42,43,3,201,5,6]): #41,42,43,101,3,201,5,6 
    station_name = stations_dic_rev[wflow_id]
    df_ = runs[r]["res"]["Q_gauges_Sall"].sel(index=wflow_id , time=slice(start, stop)).to_dataframe()["Q_gauges_Sall"]
    df_ = df_.rename(str(station_name)) #index={f"{var}_geul_stations":wflow_id})
    df_c = pd.concat([df_c,df_], axis=1)

# fig, ax = plt.subplots()    
df_c.plot.area(ax=ax2)
runs[r]["res"]["Q_gauges_Sall"].sel(index=4 , time=slice(start, stop)).to_dataframe()["Q_gauges_Sall"].plot(ax=ax2, linestyle="--", color="k", label = "Chooz")
ax2.set_ylim([0,3500])
ax2.set_xlabel("")
# ax2.set_ylabel("Q (m$^{3} s^{-1}$)", fontsize = fs)
ax2.set_title("Main contributions for the Meuse at Chooz", fontsize = fs)
ax2.tick_params(axis = "both", labelsize = fs)
ax2.legend(fontsize = 5)

plt.tight_layout()
if r == "sbm_regnie":
    plt.savefig(os.path.join(Folder_plots, "July2021_stacked_flow_borgh_chooz_regnie.png"), dpi=300)
else:
    plt.savefig(os.path.join(Folder_plots, "July2021_stacked_flow_borgh_chooz_radar.png"), dpi=300)

#%%

#Amay and ardenne peak

dic_plots = {"kin_lociner": {"run1":"kinematic",
                       "run2": "loc.iner"},

             "lociner_flpl1d_1d2d": {"run1":"loc.iner.flpl1d",
                       "run2": "loc.iner1d2d"},
                       
                       }
for dic_plot in dic_plots:
    fig, (ax1,ax2) = plt.subplots(1,2, figsize = (16/2.54, 8/2.54), sharex=True, sharey=True)
    r = dic_plots[dic_plot]["run1"]
    runs[r]["res"]["Q_gauges_Sall"].sel(index=4 , time=slice(start, stop)).to_dataframe()["Q_gauges_Sall"].plot(ax=ax1, linestyle=":", color="orange", label = "Chooz")
    runs[r]["res"]["Q_gauges_Sall"].sel(index=[10,11,12] , time=slice(start, stop)).sum("index").to_dataframe()["Q_gauges_Sall"].plot(ax=ax1, linestyle="-", color="g", label = "Ardennes")
    runs[r]["res"]["Q_gauges_Sall"].sel(index=1401 , time=slice(start, stop)).to_dataframe()["Q_gauges_Sall"].plot(ax=ax1, linestyle="-", color="r", label = "Amay")
    runs[r]["res"]["Q_gauges_Sall"].sel(index=16 , time=slice(start, stop)).to_dataframe()["Q_gauges_Sall"].plot(ax=ax1, linestyle="-", color="darkblue", label = "Borgharen")
    #add obs
    wflow_id = 16
    ds_obs_fews = df_obs_fews[wflow_id].to_xarray()
    ds_obs_fews.name = str(wflow_id)
    ds_obs_fews.sel(index=slice(start,stop)).to_dataframe()[f"{wflow_id}"].plot(ax=ax1, color = "k", label = "Borgharen obs.", linewidth = 0.8, linestyle = "--")
    ax1.set_title(f"{r}", fontsize = fs)
    ax1.set_xlabel("")
    ax1.set_ylabel("Q (m$^{3} s^{-1}$)", fontsize = fs)
    ax1.tick_params(axis = "both", labelsize = fs)
    ax1.legend(fontsize = 8)
    r2 = dic_plots[dic_plot]["run2"]
    runs[r2]["res"]["Q_gauges_Sall"].sel(index=4 , time=slice(start, stop)).to_dataframe()["Q_gauges_Sall"].plot(ax=ax2, linestyle=":", color="orange", label = "Chooz")
    runs[r2]["res"]["Q_gauges_Sall"].sel(index=[10,11,12] , time=slice(start, stop)).sum("index").to_dataframe()["Q_gauges_Sall"].plot(ax=ax2, linestyle="-", color="g", label = "Ardennes")
    runs[r2]["res"]["Q_gauges_Sall"].sel(index=1401 , time=slice(start, stop)).to_dataframe()["Q_gauges_Sall"].plot(ax=ax2, linestyle="-", color="r", label = "Amay")
    runs[r2]["res"]["Q_gauges_Sall"].sel(index=16 , time=slice(start, stop)).to_dataframe()["Q_gauges_Sall"].plot(ax=ax2, linestyle="-", color="darkblue", label = "Borgharen")
    #add obs
    wflow_id = 16
    ds_obs_fews = df_obs_fews[wflow_id].to_xarray()
    ds_obs_fews.name = str(wflow_id)
    ds_obs_fews.sel(index=slice(start,stop)).to_dataframe()[f"{wflow_id}"].plot(ax=ax2, color = "k", label = "Borgharen obs.", linewidth = 0.8, linestyle = "--")
    ax2.set_title(f"{r2}", fontsize = fs)
    ax2.tick_params(axis = "both", labelsize = fs)
    ax2.set_xlabel("")
    # ax1.set_ylabel("Q (m$^{3} s^{-1}$)", fontsize = fs)
    ax2.set_ylim([0,3500])
    ax2.legend(fontsize = 8)
    plt.tight_layout()
    plt.savefig(os.path.join(Folder_plots, f"July2021_Amay_Ardennes_Borgharen_{dic_plot}.png"), dpi=300)

#%%
# Can I delete this cell? It seems to be the same as the previous one
#Amay and ardenne peak with Nriv change
fig, (ax1,ax2) = plt.subplots(1,2, figsize = (16/2.54, 6/2.54), sharex=True, sharey=True)
r = "sbm_regnie"
runs[r]["res"]["Q_gauges_Sall"].sel(index=4 , time=slice(start, stop)).to_dataframe()["Q_gauges_Sall"].plot(ax=ax1, linestyle=":", color="orange", label = "Chooz")
runs[r]["res"]["Q_gauges_Sall"].sel(index=[10,11,12] , time=slice(start, stop)).sum("index").to_dataframe()["Q_gauges_Sall"].plot(ax=ax1, linestyle="-", color="r", label = "Ardennes")
runs[r]["res"]["Q_gauges_Sall"].sel(index=1401 , time=slice(start, stop)).to_dataframe()["Q_gauges_Sall"].plot(ax=ax1, linestyle="-", color="g", label = "Amay")
runs[r]["res"]["Q_gauges_Sall"].sel(index=16 , time=slice(start, stop)).to_dataframe()["Q_gauges_Sall"].plot(ax=ax1, linestyle="--", color="k", label = "Borgharen")
ax1.set_title("Regnie", fontsize = fs)
ax1.set_xlabel("")
ax1.set_ylabel("Q (m$^{3} s^{-1}$)", fontsize = fs)
ax1.tick_params(axis = "both", labelsize = fs)
ax1.legend(fontsize = 8)
r = "sbm_radar"
runs[r]["res"]["Q_gauges_Sall"].sel(index=4 , time=slice(start, stop)).to_dataframe()["Q_gauges_Sall"].plot(ax=ax2, linestyle=":", color="orange", label = "Chooz")
runs[r]["res"]["Q_gauges_Sall"].sel(index=[10,11,12] , time=slice(start, stop)).sum("index").to_dataframe()["Q_gauges_Sall"].plot(ax=ax2, linestyle="-", color="r", label = "Ardennes")
runs[r]["res"]["Q_gauges_Sall"].sel(index=1401 , time=slice(start, stop)).to_dataframe()["Q_gauges_Sall"].plot(ax=ax2, linestyle="-", color="g", label = "Amay")
runs[r]["res"]["Q_gauges_Sall"].sel(index=16 , time=slice(start, stop)).to_dataframe()["Q_gauges_Sall"].plot(ax=ax2, linestyle="--", color="k", label = "Borgharen")
ax2.set_title("Radar", fontsize = fs)
ax2.tick_params(axis = "both", labelsize = fs)
ax2.set_xlabel("")
# ax1.set_ylabel("Q (m$^{3} s^{-1}$)", fontsize = fs)
# ax2.legend(fontsize = 8)
# ax1.set_ylabel("Q (m$^{3} s^{-1}$)", fontsize = fs)
# ax3.legend(fontsize = 8)
plt.tight_layout()
plt.savefig(os.path.join(Folder_plots, "July2021_Amay_Ardennes_Borgharen_3runs.png"), dpi=300)


#%%
# Why this cell?
r = "sbm_regnie"
fig, ax2 = plt.subplots()
runs[r]["res"]["Q_gauges_Sall"].sel(index=12 , time=slice(start, stop)).to_dataframe()["Q_gauges_Sall"].plot(ax=ax2, linestyle="--", color="k", label = "Chooz")
runs[r]["res"]["Q_gauges_Sall"].sel(index=9 , time=slice(start, stop)).to_dataframe()["Q_gauges_Sall"].plot(ax=ax2, linestyle="--", color="r", label = "Chooz")
runs[r]["res"]["Q_gauges_Sall"].sel(index=201 , time=slice(start, stop)).to_dataframe()["Q_gauges_Sall"].plot(ax=ax2, linestyle="--", color="y", label = "Chooz")
runs[r]["res"]["Q_gauges_Sall"].sel(index=16 , time=slice(start, stop)).to_dataframe()["Q_gauges_Sall"].plot(ax=ax2, linestyle="--", color="b", label = "Chooz")


#%%
# same as three cells up but for radar

r = "sbm_radar"
#boven borgharen
df_c = pd.DataFrame()
for i, wflow_id in enumerate([ 1401, 11, 12, 10, ]): #41,42,43,101,3,201,5,6 
    station_name = stations_dic_rev[wflow_id]
    df_ = runs[r]["res"]["Q_gauges_Sall"].sel(index=wflow_id , time=slice(start, stop)).to_dataframe()["Q_gauges_Sall"]
    df_ = df_.rename(str(station_name)) #index={f"{var}_geul_stations":wflow_id})
    df_c = pd.concat([df_c,df_], axis=1)

fig, (ax1,ax2) = plt.subplots(1,2, figsize = (16/2.54, 10/2.54))    
df_c.plot.area(ax=ax1)
runs[r]["res"]["Q_gauges_Sall"].sel(index=16 , time=slice(start, stop)).to_dataframe()["Q_gauges_Sall"].plot(ax=ax1, linestyle="--", color="k", label = "Borgharen")
ax1.set_ylim([0,3500])
ax1.set_ylabel("Q (m$^{3} s^{-1}$)", fontsize = fs)
ax1.set_title("Main contributions for the Meuse at Borgharen", fontsize = fs)
# plt.savefig(os.path.join(Folder_plots, "July2021_stacked_flow_chooz.png"), dpi=300)
ax1.tick_params(axis = "both", labelsize = fs)
ax1.legend(fontsize = 5)
ax1.set_xlabel("")

#boven chooz
df_c = pd.DataFrame()
for i, wflow_id in enumerate([ 41,42,43,3,201,5,6]): #41,42,43,101,3,201,5,6 
    station_name = stations_dic_rev[wflow_id]
    df_ = runs[r]["res"]["Q_gauges_Sall"].sel(index=wflow_id , time=slice(start, stop)).to_dataframe()["Q_gauges_Sall"]
    df_ = df_.rename(str(station_name)) #index={f"{var}_geul_stations":wflow_id})
    df_c = pd.concat([df_c,df_], axis=1)

# fig, ax = plt.subplots()    
df_c.plot.area(ax=ax2)
runs[r]["res"]["Q_gauges_Sall"].sel(index=4 , time=slice(start, stop)).to_dataframe()["Q_gauges_Sall"].plot(ax=ax2, linestyle="--", color="k", label = "Chooz")
ax2.set_ylim([0,3500])
ax2.set_xlabel("")
# ax2.set_ylabel("Q (m$^{3} s^{-1}$)", fontsize = fs)
ax2.set_title("Main contributions for the Meuse at Chooz", fontsize = fs)
ax2.tick_params(axis = "both", labelsize = fs)
ax2.legend(fontsize = 5)

plt.tight_layout()
if r == "sbm_regnie":
    plt.savefig(os.path.join(Folder_plots, "July2021_stacked_flow_borgh_chooz_regnie.png"), dpi=300)
else:
    plt.savefig(os.path.join(Folder_plots, "July2021_stacked_flow_borgh_chooz_radar.png"), dpi=300)


#%% 

plt.figure(); qobs_h_fr["Q"].sel(wflow_id=[1011, 101, 3, 4]).plot(hue = "wflow_id")


#%% quick check N 0.01
kwargs = {"index_col":0, "parse_dates":True, "header":0}
output_regnie = pd.read_csv(r"p:\11208719-interreg\wflow\wflow_meuse_julia_2021\run_july_2021_oper_sbm_lociner2d_radar\run_july2021_regnie_all\output.csv", **kwargs)
output_radar = pd.read_csv(r"p:\11208719-interreg\wflow\wflow_meuse_julia_2021\run_july_2021_oper_sbm_lociner2d_radar\run_july2021_radar\output.csv", **kwargs)

catch = 10 #4
fig, ax = plt.subplots()
output_regnie[f"Q_{catch}"].plot(label = "regnie")
output_radar[f"Q_{catch}"].plot(label = "radar")

plt.legend()
