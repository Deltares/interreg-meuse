#%%
import os
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt

import hydromt
from hydromt_wflow import WflowModel
#from hydromt_wflow.utils import read_csv_results
from hydromt.stats import skills

from plots import plot_model_results, plot_additional_sign

#%% Read the data. SnakeMake arguments are automatically passed

csv_fns = snakemake.input.csv_files
toml_default_fn = snakemake.params.toml_default
obs_catalog = snakemake.params.obs_catalog
obs_fn = snakemake.params.obs_fn
obs_ts_fn = snakemake.params.obs_ts_fn

# cal_period = snakemake.params.calibration_period
# eval_period = snakemake.params.evaluation_period
cal_start = snakemake.params.cal_start
cal_end = snakemake.params.cal_end
eval_start = snakemake.params.eval_start
eval_end = snakemake.params.eval_end

# Efficiency criteria to use for selection
# EFF_OPT = "KGE_np_flood"
# EFF_OPT = "dist_nse_nselog"
EFF_OPT = "dist_nse_nselog_nsenm7q"


#%% Helper functions

def compute_average_upstream(params, flwdir, ds_like, mask=None):
    """
    Compute parameters maps with upstream average values of the parameters for every cells.

    Parameters
    ----------
    params: list of str
        List of parameters name in ds_like (can be 'param' or 'param1-param2')
    ds_like: xarray.Dataset
        Dataset at model resolution
    flwdir: pyflwdir.FlwdirRaster
        Flow direction raster object
    mask: xarray.DataArray, optional
        Mask grid cells outside of domain.
    
    Returns
    -------
    ds_out : xarray.Dataset
        Dataset containing the upstream parameters maps
    
    """
    for p in params:
        if len(p.split('-')) > 1:
            p1 = p.split('-')[0]
            p2 = p.split('-')[1]
            paracc = flwdir.accuflux(ds_like[p1].values - ds_like[p2].values)
            nodata = ds_like[p1].raster.nodata
        else:
            paracc = flwdir.accuflux(ds_like[p].values)
            nodata = ds_like[p].raster.nodata
        if mask is not None:
            cells = xr.where(mask != mask.raster.nodata, mask * 0 + 1, mask.raster.nodata)
        else:
            cells = paracc.copy() *0 + 1
        cellsacc = flwdir.accuflux(cells.values)
        par_mean_upstream = np.where(cellsacc > 0, paracc/cellsacc, paracc)
        if p == params[0]:
            ds_out = xr.DataArray(
                name=f"{p}_upstream_mean",
                data= par_mean_upstream,
                coords = ds_like.raster.coords,
                dims=ds_like.raster.dims,
                attrs = dict(
                    long_name=f'Average upstream {p} value',
                    _FillValue = nodata,
                )
            ).to_dataset()
        else:
            da = xr.DataArray(
                name=f"{p}_upstream_mean",
                data= par_mean_upstream,
                coords = ds_like.raster.coords,
                dims=ds_like.raster.dims,
                attrs = dict(
                    long_name=f'Average upstream {p} value',
                    _FillValue = nodata,
                )
            )
            ds_out[f"{p}_upstream_mean"] = da
    
    return ds_out


#%%

# Instantiate wflow model
print("Reading wflow model")
root = os.path.dirname(toml_default_fn)
mod = WflowModel(root, config_fn=os.path.basename(toml_default_fn), data_libs = obs_catalog, mode="r")

# Read gauges staticgeoms and get snapped coordinates of the stations
gdf_gauges = mod.staticgeoms[f"gauges_{obs_fn}"]
if obs_fn == "grdc":
    gdf_gauges = gdf_gauges.set_index("grdc_no")
elif obs_fn == "gb-camels":
    gdf_gauges = gdf_gauges.set_index("gauge_id")
elif obs_fn == "stations_obs":
    gdf_gauges = gdf_gauges.set_index("id")
else:
    gdf_gauges = gdf_gauges.set_index("wflow_id")
# Filter gauges that were not snapped (was fixed in a later hydromt version)
#da_gauges = mod.staticmaps[f'wflow_gauges_{obs_fn}']
#ids_gauges = np.unique(da_gauges.values[da_gauges.values != da_gauges.raster.nodata])
#gdf_gauges = gdf_gauges[gdf_gauges.index.isin(ids_gauges)]
xs, ys = np.vectorize(lambda p: (p.xy[0][0], p.xy[1][0]))(gdf_gauges["geometry"])
idxs_gauges = mod.staticmaps.raster.xy_to_idx(xs, ys)

outdir = os.path.join(os.path.dirname(csv_fns[0]), "Results", "Per_station")
if not os.path.isdir(outdir):
    os.makedirs(outdir)

#%% Read observations
print("Reading observations")
obs = mod.data_catalog.get_geodataset(obs_ts_fn, geom=mod.basins)
obs = obs.vector.to_crs(mod.crs)

# Get start and endtime
# starttime = max(obs.time[0], pd.to_datetime(mod.get_config("starttime")))
# endtime = min(obs.time[-1], pd.to_datetime(mod.get_config("endtime")))
starttime = cal_start
endtime = eval_end

obs = obs.sel(time=slice(starttime, endtime)).load()

#%% Create dict with the different calibration runs
nb_runs = len(csv_fns)
runs = dict()

KsH_values = [int(os.path.basename(i).split(".")[0].split("~")[1]) for i in csv_fns]

print("Reading results for the different runs")
# import pdb; pdb.set_trace()
for i in range(nb_runs): #range(2):
    longname = f"KsH{KsH_values[i]}"
    df = pd.read_csv(csv_fns[i], index_col=0, parse_dates=True, header=0)
    ## ds = xr.open_dataset(nc_fns[i]).rename({f'Q_gauges_{obs_fn}': 'index'}) 
    ## ds['index'] = ds['index'].astype(np.int32)
    ri = {
        'longname': longname,
        ## 'results': ds.load(),
        'results': df,
        'KsatHorFrac': KsH_values[i] 
    }
    runs[f"run{i}"] = ri

#%% Compute maps with upstream average Ksatver / f / (thetaS - thetaR) values
ds_up_avg = compute_average_upstream(
    params = ['KsatVer', 'f', 'thetaS', 'thetaR', 'thetaS-thetaR'],
    flwdir = mod.flwdir,
    ds_like = mod.staticmaps,
    mask = mod.staticmaps["wflow_subcatch"],
)

# Instantiate the NetCDF output files
# File 1: Properties per station and efficiency scores of the best run
ds_out = xr.Dataset(
    data_vars=dict(
        uparea=(["index"], mod.staticmaps['wflow_uparea'].values.flat[idxs_gauges]),
        ksatver=(["index"], ds_up_avg['KsatVer_upstream_mean'].values.flat[idxs_gauges]),
        f=(["index"], ds_up_avg['f_upstream_mean'].values.flat[idxs_gauges]),
        thetas=(["index"], ds_up_avg['thetaS_upstream_mean'].values.flat[idxs_gauges]),
        thetadiff=(["index"], ds_up_avg['thetaS-thetaR_upstream_mean'].values.flat[idxs_gauges]),
        ksathorfrac=(["index"], np.zeros(len(gdf_gauges.index.values))),
        kge_np_flood=(["index"], np.zeros(len(gdf_gauges.index.values))),
        nse=(["index"], np.zeros(len(gdf_gauges.index.values))),
        kge_2009=(["index"], np.zeros(len(gdf_gauges.index.values))),
        biais=(["index"], np.zeros(len(gdf_gauges.index.values))),
        nse_log=(["index"], np.zeros(len(gdf_gauges.index.values))),
        dist_nse_nselog=(["index"], np.zeros(len(gdf_gauges.index.values))),
        dist_nse_nselog_nsenm7q=(["index"], np.zeros(len(gdf_gauges.index.values))),
    ),
    coords=dict(
        index = gdf_gauges.index.values.astype('int32'),
        x = (["index"], xs),
        y = (["index"], ys),
    ),
    attrs = dict(description=f"Model performance and hydrological properties of wflow model at the {obs_fn} stations.")
)
# Attributes
ds_out.raster.set_crs(mod.crs)
ds_out["uparea"].raster.set_nodata(mod.staticmaps['wflow_uparea'].raster.nodata)
ds_out["ksatver"].raster.set_nodata(ds_up_avg['KsatVer_upstream_mean'].raster.nodata)
ds_out["f"].raster.set_nodata(ds_up_avg['f_upstream_mean'].raster.nodata)
ds_out["thetas"].raster.set_nodata(ds_up_avg['thetaS_upstream_mean'].raster.nodata)
ds_out["thetadiff"].raster.set_nodata(ds_up_avg['thetaS-thetaR_upstream_mean'].raster.nodata)

# File 2: Efficiency scores for all runs per station
ds_eff = xr.Dataset(
    data_vars=dict(
        kge_np_flood=(["index", "ksathorfrac"], np.zeros((len(gdf_gauges.index.values), len(KsH_values)))),
        nse=(["index", "ksathorfrac"], np.zeros((len(gdf_gauges.index.values), len(KsH_values)))),
        kge_2009=(["index", "ksathorfrac"], np.zeros((len(gdf_gauges.index.values), len(KsH_values)))),
        biais=(["index", "ksathorfrac"], np.zeros((len(gdf_gauges.index.values), len(KsH_values)))),
        nse_log=(["index", "ksathorfrac"], np.zeros((len(gdf_gauges.index.values), len(KsH_values)))),
        dist_nse_nselog=(["index", "ksathorfrac"], np.zeros((len(gdf_gauges.index.values), len(KsH_values)))),
        dist_nse_nselog_nsenm7q =(["index", "ksathorfrac"], np.zeros((len(gdf_gauges.index.values), len(KsH_values)))),
    ),
    coords=dict(
        index = gdf_gauges.index.values.astype('int32'),
        ksathorfrac= KsH_values,
        x = (["index"], xs),
        y = (["index"], ys),
    ),
)
ds_eff.raster.set_crs(mod.crs)

#%% Loop over the stations to get station properties, performance and plots
# Save results as a dictionary per station
print("Analysing results at the different stations")
#stations = dict()
for i in range(len(gdf_gauges.index)):
    st = gdf_gauges.index.values[i]
    if st in obs.index:
        print(f"Station {st} ({i+1}/{len(gdf_gauges.index)})")

        #name of discharge in obs netcdf and in csv !
        discharge_name = "Q"

        #check if there is observed data
        # Read observation at the station
        obs_i = obs.sel(index=st)
        mask = ~obs_i.isnull()
        try:
            obs_i_nan = obs_i.where(mask, drop = True)
        except:
            obs_i_nan = obs_i[discharge_name].where(mask[discharge_name], drop = True)
        if len(obs_i_nan.time) > 2*366: # should be changed if hourly! make sure enough observed data length   
        # plot the different runs and compute performance
            performance = plot_model_results(obs, runs, station=st, discharge_name=discharge_name, savedir=outdir)
            performance.index = KsH_values
            performance.index.name = "KsatHorFrac"

            #plot additional signatures
            plot_additional_sign(obs, runs, station=st, discharge_name="Q", savedir=outdir, performance=performance)

            # add performance to netdcf file 2
            ds_eff["kge_np_flood"].loc[dict(index=st)] = performance["KGE_np_flood"]
            ds_eff["nse"].loc[dict(index=st)] = performance["NSE"]
            ds_eff["kge_2009"].loc[dict(index=st)] = performance["KGE_2009"]
            ds_eff["biais"].loc[dict(index=st)] = performance["Biais"]
            ds_eff["nse_log"].loc[dict(index=st)] = performance["NSE_log"]
            ds_eff["dist_nse_nselog"].loc[dict(index=st)] = performance["dist_nse_nselog"]
            ds_eff["dist_nse_nselog_nsenm7q"].loc[dict(index=st)] = performance["dist_nse_nselog_nsenm7q"]

            # Get optimal ksathorfrac value based on the chosen efficiency criteria (EFF_OPT)
            if EFF_OPT in ["dist_nse_nselog", "dist_nse_nselog_nsenm7q"]:
                eff_opt = np.nanmin(performance[EFF_OPT])
            else:
                eff_opt = np.nanmax(performance[EFF_OPT])
            # Check if station has observations
            if not np.isnan(eff_opt):
                ds_out["ksathorfrac"].loc[dict(index=st)] = performance.index.values[performance[EFF_OPT] == eff_opt][0]
                ds_out["kge_np_flood"].loc[dict(index=st)] = performance["KGE_np_flood"].values[performance[EFF_OPT] == eff_opt][0]
                ds_out["kge_2009"].loc[dict(index=st)] = performance["KGE_2009"].values[performance[EFF_OPT] == eff_opt][0]
                ds_out["nse"].loc[dict(index=st)] = performance["NSE"].values[performance[EFF_OPT] == eff_opt][0]
                ds_out["biais"].loc[dict(index=st)] = performance["Biais"].values[performance[EFF_OPT] == eff_opt][0]
                ds_out["nse_log"].loc[dict(index=st)] = performance["NSE_log"].values[performance[EFF_OPT] == eff_opt][0]
                ds_out["dist_nse_nselog"].loc[dict(index=st)] = performance["dist_nse_nselog"].values[performance[EFF_OPT] == eff_opt][0]
                ds_out["dist_nse_nselog_nsenm7q"].loc[dict(index=st)] = performance["dist_nse_nselog_nsenm7q"].values[performance[EFF_OPT] == eff_opt][0]
            else:
                ds_out["ksathorfrac"].loc[dict(index=st)] = -9999.
                ds_out["kge_np_flood"].loc[dict(index=st)] = -9999.
                ds_out["kge_2009"].loc[dict(index=st)] = -9999.
                ds_out["nse"].loc[dict(index=st)] = -9999.
                ds_out["biais"].loc[dict(index=st)] = -9999.
                ds_out["nse_log"].loc[dict(index=st)] = -9999.
                ds_out["dist_nse_nselog"].loc[dict(index=st)] = -9999.
                ds_out["dist_nse_nselog_nsenm7q"].loc[dict(index=st)] = -9999.


#%% Save the netcdf files

# Filter and update the fill values
vars = ["ksathorfrac", "kge_np_flood", "kge_2009", "nse", "biais", "nse_log", "dist_nse_nselog", "dist_nse_nselog_nsenm7q"]
for v in vars:
    ds_out[v].raster.set_nodata(-9999.)
    if v != "ksathorfrac":
        ds_eff[v] = xr.where(ds_eff[v].isnull, ds_eff[v], -9999.)
        ds_eff[v].raster.set_nodata(-9999.)

# Write to netcdf
nc_fn1 = os.path.join(outdir, "..", f"stations_analysis.nc")
ds_out.to_netcdf(nc_fn1)

nc_fn2 = os.path.join(outdir, "..", f"stations_performance.nc")
ds_eff.to_netcdf(nc_fn2)