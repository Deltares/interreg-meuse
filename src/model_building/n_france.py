import hydromt
import xarray as xr
import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
from hydromt_wflow import WflowModel
import glob

pdir = r"p:\11208719-interreg\wflow"
case_base = "m_snakecal03"
case_update = "n_france"
run_folder = "run_snakecal03"
config_fn = "run_snakecal03.toml"
run_folder_update = "run_france"

root = os.path.join(pdir, case_base, run_folder)
mod = WflowModel(root=root, config_fn=config_fn, mode="r")

#initialize staticmaps and staticgeoms
mod.staticmaps
mod.staticgeoms


#based on grade data files - update the ids of geojson and netcdf and write to interreg p drive
gauges_path = r"p:\11208719-interreg\data\observed_streamflow_grade\FR-Hydro-Meuse.geojson"
#make wflow id match real hydrobanque id 
gauges = gpd.read_file(gauges_path)
#old ids of phd work
gauges = gauges.rename(columns={"wflow_id":"wflow_id_old"})
#new ids based on real data
gauges["wflow_id"] = gauges["index"].str.replace("B", "1").astype("int64")

#write adapted geojson france to folder data/france
gauges.to_file(r"p:\11208719-interreg\data\france\c_final\france.geojson")
#also update the coordinates of the nc file:
nc_gauges = xr.open_dataset(r"p:\11208719-interreg\data\observed_streamflow_grade\FR-Hydro-hourly-2005_2022.nc")
nc_gauges = nc_gauges.rename({"wflow_id": "wflow_id_old"})
list_ids = list(nc_gauges.index.values)
list_ids_wflow = [int(s.replace('B', '1')) for s in list_ids]
nc_gauges = nc_gauges.assign_coords({"wflow_id" : ("wflow_id_old", list_ids_wflow)})
nc_gauges = nc_gauges.swap_dims({"wflow_id_old":"wflow_id"})
nc_gauges.to_netcdf(r"p:\11208719-interreg\data\france\c_final\france_hourly.nc")

#daily data french stations 
nc_gauges_d = xr.open_dataset(r"p:\11208719-interreg\data\observed_streamflow_grade\qobs_xr.nc")
# get french stations out and update id 
nc_gauges_d_fr = nc_gauges_d.rename({"catchments":"wflow_id_old"})
nc_gauges_d_fr = nc_gauges_d_fr.sel(wflow_id_old = gauges["wflow_id_old"].values)
nc_gauges_d_fr = nc_gauges_d_fr.assign_coords({"wflow_id" : ("wflow_id_old", gauges["wflow_id"].values)})
nc_gauges_d_fr = nc_gauges_d_fr.assign_coords({"Libellé" : ("wflow_id_old", gauges["Libellé"].values)})
nc_gauges_d_fr = nc_gauges_d_fr.assign_coords({"x" : ("wflow_id_old", gauges.geometry.x.values)})
nc_gauges_d_fr = nc_gauges_d_fr.assign_coords({"y" : ("wflow_id_old", gauges.geometry.y.values)})
nc_gauges_d_fr = nc_gauges_d_fr.swap_dims({"wflow_id_old":"wflow_id"})
#write to netcdf
nc_gauges_d_fr.to_netcdf(r"p:\11208719-interreg\data\france\c_final\france_daily.nc")

#and get borgharen out. 
nc_gauges_d_rws = nc_gauges_d.sel(catchments = [16])
nc_gauges_d_rws = nc_gauges_d_rws.rename({"catchments":"wflow_id"})
nc_gauges_d_rws = nc_gauges_d_rws.assign_coords({"station_name" : ("wflow_id", ["Meuse at St Pieter"])})
nc_gauges_d_rws = nc_gauges_d_rws.assign_coords({"x" : ("wflow_id", [5.69554])})
nc_gauges_d_rws = nc_gauges_d_rws.assign_coords({"y" : ("wflow_id", [50.8485])})
#write to netcdf
nc_gauges_d_rws.to_netcdf(r"p:\11208719-interreg\data\rwsinfo\c_final\rwsinfo_daily.nc")

# st pieter hourly to netcdf 
stpieter = pd.read_csv(r"p:\11208719-interreg\data\observed_streamflow_grade\20221205_ST_PIETER\05_OUTPUT\ST_PIETER.csv", index_col=0, parse_dates=True, header=0)
stpieter = stpieter.resample("H").mean()
stpieter_xr = stpieter.to_xarray()
stpieter_xr = stpieter_xr.rename({"value":"Q"})
stpieter_xr = stpieter_xr.rename({"timestamp":"time"})
stpieter_xr = stpieter_xr.assign_coords({"wflow_id": 16}).expand_dims("wflow_id")
stpieter_xr = stpieter_xr.assign_coords({"station_name" : ("wflow_id", ["Meuse at St Pieter"])})
stpieter_xr = stpieter_xr.assign_coords({"x" : ("wflow_id", [5.69554])})
stpieter_xr = stpieter_xr.assign_coords({"y" : ("wflow_id", [50.8485])})
#write to netcdf
stpieter_xr.to_netcdf(r"p:\11208719-interreg\data\rwsinfo\c_final\rwsinfo_hourly.nc")


#now update model 
basename = "france"
gauge_toml_header = ["Q_france"]
gauge_toml_param = ["lateral.river.q_av"]

for o in range(len(gauge_toml_param)):
    gauge_toml_dict = {
                    "header": gauge_toml_header[o],
                    "map": f"gauges_{basename}",
                    "parameter": gauge_toml_param[o],
                }
    mod.config["csv"]["column"].append(gauge_toml_dict)

mod.setup_gauges(gauges_fn=gauges, 
                 index_col="wflow_id", #make sure id is int 64 type !!!! 
                 derive_subcatch=True, 
                 basename=basename,
                 toml_output = "netcdf",
                 gauge_toml_header=gauge_toml_header, 
                 gauge_toml_param=gauge_toml_param)

root_updated = os.path.join(pdir, case_update, run_folder_update)
mod.set_root(root_updated)

mod.write_staticmaps()
mod.write_staticgeoms()
mod.write_config(config_name=f"{run_folder_update}.toml")

#copy of lake files 
lake_files = glob.glob(os.path.join(pdir, case_base, "lake*.csv"))
if len(lake_files)>0:
    for lake_file in lake_files:
        dst_lake_file = os.path.join(pdir, case_update, os.path.basename(lake_file))
        shutil.copy(lake_file, dst_lake_file)

# remove run_default dir
if len(os.listdir(os.path.join(pdir, case_update, run_folder_update, 'run_default'))) == 0:
    os.rmdir(os.path.join(pdir, case_update, run_folder_update, 'run_default'))

# remove instate dir
if len(os.listdir(os.path.join(pdir, case_update, run_folder_update, 'instate'))) == 0:
    os.rmdir(os.path.join(pdir, case_update, run_folder_update, 'instate'))

# move staticgeoms
shutil.move(os.path.join(pdir, case_update, run_folder_update, 'staticgeoms'),
            os.path.join(pdir, case_update, 'staticgeoms'),
            )

# write run bat file
path_env_julia = r"c:\Users\bouaziz\.julia\environments\wflow_floodplain1d"
# path_env_julia = r"c:\Users\riveros\.julia\environments\v1.6"

bat_str = f"""
julia  -t 4 --project={path_env_julia} -e "using Wflow; Wflow.run()" "{run_folder_update}\{run_folder_update}.toml"
"""

bat_file = os.path.join(pdir, case_update, f"{run_folder_update}.cmd")
with open(bat_file, mode="w") as f:
    f.write(bat_str)