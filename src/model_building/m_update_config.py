#%%
import xarray as xr
import os
from hydromt_wflow import WflowModel

Folder_p = r"p:\11208719-interreg\wflow\m_snakecal03"

cases = [
        "run_snakecal03_july_2021",
        "run_snakecal03_july_2021_1d",
        "run_snakecal03_july_2021_1d2d",
        "run_snakecal03_july_2021_kinematic",
]

config_fns = [
                "run_snakecal03_july_2021_regnie.toml", 
                "run_snakecal03_july_2021_era5.toml"
]

for case in cases:
    for config_fn in config_fns:
        print(case, config_fn)

        root = os.path.join(Folder_p, case)
        mod = WflowModel(root=root, config_fn = config_fn, mode ="r+")

        #update config after calibration 
        mod.config
        mod.set_config("input.vertical.rootingdepth", "RootingDepth_obs_20_cal")
        mod.set_config("input.vertical.soilminthickness", "SoilThickness_cal")
        mod.set_config("input.vertical.soilthickness", "SoilThickness_cal")
        mod.set_config("input.vertical.storage_wood", "Swood_cal")
        mod.set_config("input.vertical.maxleakage", "MaxLeakage_cal")

        mod.set_config("input.lateral.river.n", "N_River_cal")

        mod.set_config("input.lateral.land.n", "N_cal")

        mod.set_config("input.lateral.subsurface.ksathorfrac", "ksathorfrac_sub_cal")

        mod.write_config()


        

#%%