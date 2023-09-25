#%%
import xarray as xr
import numpy as np
import os
import hydromt
#%%
fn_in = r'/p/11208719-interreg/data/racmo/members_bias_corrected_revised/c_wflow/daily'
members = [f'r{int(i)}i1p5f1' for i in np.arange(1,17,1)]
# %%
#encoding otherwise wflow doesn't run
chunksizes = (1, ds_merge.lat.size, ds_merge.lon.size)
encoding = {
        v: {"zlib": True, "dtype": "float32", "chunksizes": chunksizes}
        for v in ds_merge.data_vars.keys()
    }
encoding["time"] = {"_FillValue": None}

for member in members:
    print(member)
    all_rainfalls = list()
    for year in np.arange(1950,1955,1):
        print(year)
        ds = xr.open_dataset(os.path.join(fn_in, member, f'ds_merged_{year}.nc'))
        all_rainfalls.append(ds['precip'])
    # Concatenate the datasets along the time dimension
    ds_member_precip = xr.concat(all_rainfalls, dim="time")

#%%

mask = ds_memeber_precip.raster.geometry_mask(max)
# %%
