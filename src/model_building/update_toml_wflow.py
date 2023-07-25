from hydromt_wflow import WflowModel
from hydromt import DataCatalog
import pandas as pd
import numpy as np
import os, glob, sys
import xarray as xr

#We read the snakemake parameters
starttime = snakemake.params.wflow_params["starttime"]
endtime = snakemake.params.wflow_params["endtime"]
timestep = snakemake.params.timestep
exp_name = snakemake.params.exp_name
model = snakemake.params.model
fn_forcing = snakemake.params.fn_forcing
fn = snakemake.params.fn_wflow
member_nb = snakemake.params.member_nb
base_model_toml = snakemake.params.wflow_base_toml
start_path = snakemake.params.start_path
fn_ds = str(snakemake.input.fn_in)
fn_path_input = snakemake.params.fn_path_input
# conv_params = snakemake.params.conv_params
# fn_orography = snakemake.params.fn_orography
# fn_idx = snakemake.params.fn_idx

print("------- Checking what we got ------")
print("starttime", starttime)
print("endtime", endtime)
print("exp_name", exp_name)
print("model", model)
print("fn_forcing", fn_forcing)
print("fn", fn)
print("dt", timestep)
print("member_nb", member_nb)
print("base_model_toml", base_model_toml)
print("current path", start_path)
print("ds path", fn_ds)
print("fn_path_input", fn_path_input)
# print("fn_orography", fn_orography)
# print("fn_idx", fn_idx)

#%% 
#Getting the frequency from the netcdf file
ds = xr.open_dataset(os.path.join(fn_ds))
freq = (ds.time[1].values-ds.time[0].values)/np.timedelta64(1, 's')
ds.close()

#Setting the locations of the model to read it
fn_static = os.path.abspath(os.path.join(fn, model, 'staticmaps.nc'))
fn_forcing_all = os.path.abspath(os.path.join(fn_forcing, '*.nc'))
fn_state_output = os.path.abspath(os.path.join(fn, model, exp_name+'_'+timestep, member_nb,'outstate','outstates.nc'))
fn_csv = os.path.abspath(os.path.join(fn, model, exp_name+'_'+timestep, member_nb, 'output.csv'))
fn_log = os.path.abspath(os.path.join(fn, model, exp_name+'_'+timestep, member_nb, 'log.txt'))
fn_path_input = os.path.abspath(os.path.join(fn_path_input))

# fn_static = os.path.relpath(fn_static, start_path)
# fn_forcing_all = os.path.relpath(fn_forcing_all, start_path)
# fn_state_output = os.path.relpath(fn_state_output, start_path)
# fn_csv = os.path.relpath(fn_csv, start_path)
# fn_log = os.path.relpath(fn_log, start_path)

print("fn_static", fn_static)
print("fn_forcing_all", fn_forcing_all)

#%%We read the base model and update the parameters for our run
mod = WflowModel(root=os.path.join(fn, model), mode="r", config_fn = os.path.join(base_model_toml)) #r'p:\11208719-interreg\wflow\wflow_meuse_julia\hydromt_data.yml'
print("Base model is ", os.path.join(fn, model))

mod.set_config("timestepsecs", int(freq))
mod.set_config("starttime", starttime)
mod.set_config("endtime", endtime)
mod.set_config("input.path_static", f"{fn_static}")
mod.set_config("input.path_forcing", f"{fn_forcing_all}")
mod.set_config("state.path_output", f"{fn_state_output}")  #outstate

# mod.set_config("forcing.path_orography", f"{fn_orography}")
# mod.set_config("forcing.path_idx", f"{fn_idx}")

# # Now we instead adapt the netcdf with the arguments 
#  #"scale":2, "offset": 0})

# if conv_params["precip"]['conversion'] == 'additive':
#         mod.set_config("input.vertical.precipitation", {'netcdf':{'variable': {'name' :conv_params["precip"]["wflow_name"]}}, 'scale':1, 'offset': conv_params["precip"][timestep] }) #"scale":2, "offset": 0})
# if conv_params["precip"]['conversion'] == 'multiplicative':
#         mod.set_config("input.vertical.precipitation", {'netcdf':{'variable': {'name' :conv_params["precip"]["wflow_name"]}}, 'scale':conv_params["precip"][timestep], 'offset': 0 }) #"scale":2, "offset": 0})

# if conv_params["t2m"]['conversion'] == 'additive':
#         mod.set_config("input.vertical.temperature", {'netcdf':{'variable': {'name' :conv_params["t2m"]["wflow_name"]}}, 'scale':1, 'offset': conv_params["t2m"][timestep] }) #"scale":2, "offset": 0})
# if conv_params["t2m"]['conversion'] == 'multiplicative':
#         mod.set_config("input.vertical.temperature", {'netcdf':{'variable': {'name' :conv_params["precip"]["wflow_name"]}}, 'scale':conv_params["t2m"][timestep], 'offset': 0 }) #"scale":2, "offset": 0})

# if conv_params["pet"]['conversion'] == 'additive':
#         mod.set_config("input.vertical.potential_evaporation", {'netcdf':{'variable': {'name' :conv_params["pet"]["wflow_name"]}}, 'scale':1, 'offset': conv_params["pet"][timestep] }) #"scale":2, "offset": 0})
# if conv_params["pet"]['conversion'] == 'multiplicative':
#         mod.set_config("input.vertical.potential_evaporation", {'netcdf':{'variable': {'name' :conv_params["pet"]["wflow_name"]}}, 'scale':conv_params["pet"][timestep], 'offset': 0 }) #"scale":2, "offset": 0})

mod.set_config("input.vertical.precipitation", "precip")
mod.set_config("input.vertical.temperature", "temp")
mod.set_config("input.vertical.potential_evaporation", "pet")

mod.set_config("csv.path", f"{fn_csv}")
mod.set_config("path_log", f"{fn_log}")

mod.set_config("model.reinit", "false")
mod.set_config("state.path_input", f"{fn_path_input}")

if timestep == "daily":
        #We drop the netcdf output file
        mod.set_config("output.path", "output.nc")

#%%We write the output somewhere else
fn_output = os.path.join(fn, model, exp_name+'_'+timestep, member_nb)
mod.set_root(fn_output)
mod.write_config(config_name=f"{exp_name}_{timestep}_{member_nb}.toml")

#Delete empty folders created at the beginning
if len(os.listdir(os.path.join(fn, model, exp_name+'_'+timestep, member_nb, 'staticgeoms'))) == 0:
        os.rmdir(os.path.join(fn, model, exp_name+'_'+timestep, member_nb, 'staticgeoms'))

if len(os.listdir(os.path.join(fn, model, exp_name+'_'+timestep, member_nb, 'run_default'))) == 0:
        os.rmdir(os.path.join(fn, model, exp_name+'_'+timestep, member_nb, 'run_default'))

