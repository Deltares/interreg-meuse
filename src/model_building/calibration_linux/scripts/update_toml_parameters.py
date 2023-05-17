import os
import hydromt
from hydromt_wflow import WflowModel

#%% Read the data. SnakeMake arguments are automatically passed

paramset = snakemake.params.calibration_parameters
paramsetname = snakemake.params.calibration_pattern
toml_default_fn = snakemake.params.toml_default

# Instantiate wflow model and read/update config
root = os.path.dirname(toml_default_fn)
mod = WflowModel(root, config_fn=os.path.basename(toml_default_fn), mode="r+")

# Update parameter value in toml file
for param in paramset:
    if param == "ksathorfrac":
        #value
        # mod.set_config("input.lateral.subsurface.ksathorfrac.value", float(paramset['ksathorfrac']))
        #mult
        mod.set_config("input.lateral.subsurface.ksathorfrac.scale", float(paramset['ksathorfrac']))
    elif param == "tt" or param == "ttm" or param =="maxleakage":
        mod.set_config(f"input.vertical.{param}.value", float(paramset[param]))
    elif param == "floodplain_volume":
        mod.set_config(f"input.lateral.river.floodplain.volume.scale", float(paramset["floodplain_volume"]))
    elif param == "storage_wood":
        mod.set_config(f"input.vertical.{param}.offset", float(paramset[param]))
    elif param == "n": #land river and floodplain 
        mod.set_config(f"input.lateral.land.{param}.scale", float(paramset[param]))
        mod.set_config(f"input.lateral.river.{param}.scale", float(paramset[param]))
        mod.set_config(f"input.lateral.river.floodplain.{param}.scale", float(paramset[param]))
    else:
        mod.set_config(f"input.vertical.{param}.scale", float(paramset[param]))


# Update parameters for saving outputs
setting_toml = {
    "csv.path": f"output_{paramsetname}.csv",
    "path_log": f"log_{paramsetname}.txt"
}

for option in setting_toml:
    mod.set_config(option, setting_toml[option])

# Write new toml file
toml_root = os.path.join(os.path.dirname(toml_default_fn))
toml_name = f"wflow_sbm_{paramsetname}.toml"
mod.write_config(config_name=toml_name, config_root=toml_root)