import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os

# We read the snakemake parameters
year_random = snakemake.params.year_random
all_files = snakemake.input.fn_forcing
outfile = snakemake.output

print("------- Checking what we got ------")
print("year_random", year_random)
print("all_files[0]", all_files[0])
print("outfile[0]", outfile[0])

#We load the random file
fn = all_files[0].split("/ds_merged")[0]
fn_out = outfile[0].split("/precip_sum.png")[0]
fn_file = os.path.join(fn,f"ds_merged_{year_random}.nc")

print("------- Checking what we got ------")
print("fn", fn)
print("fn_out", fn_out)
print("fn_file", fn_file)

ds = xr.open_dataset(fn_file, chunks='auto')

#------We plot the temperature
fig, axs = plt.subplots(ncols=1, nrows=1)
ds['temp'].min(dim='time').plot(ax=axs)
fig.savefig(os.path.join(fn_out, f"temp_min_year_{year_random}.png"), dpi=300)

fig, axs = plt.subplots(ncols=1, nrows=1)
ds['temp'].max(dim='time').plot(ax=axs)
fig.savefig(os.path.join(fn_out, f"temp_max.png"), dpi=300)

fig, axs = plt.subplots(ncols=1, nrows=1)
ds['temp'].isel(lat=100, lon=100).plot(ax=axs)
fig.savefig(os.path.join(fn_out, f"temp_randomloc_year_{year_random}.png"), dpi=300)

#------We plot the precip
fig, axs = plt.subplots(ncols=1, nrows=1)
ds['precip'].sum(dim='time').plot(ax=axs)
fig.savefig(os.path.join(fn_out, f"precip_sum.png"), dpi=300)

fig, axs = plt.subplots(ncols=1, nrows=1)
ds['precip'].max(dim='time').plot(ax=axs)
fig.savefig(os.path.join(fn_out, f"precip_max_year_{year_random}.png"), dpi=300)

fig, axs = plt.subplots(ncols=1, nrows=1)
ds['precip'].isel(lat=100, lon=100).plot(ax=axs)
fig.savefig(os.path.join(fn_out, f"precip_randomloc_year_{year_random}.png"), dpi=300)

#------We plot the pet
fig, axs = plt.subplots(ncols=1, nrows=1)
ds['pet'].sum(dim='time').plot(ax=axs)
fig.savefig(os.path.join(fn_out, f"pet_sum.png"), dpi=300)

fig, axs = plt.subplots(ncols=1, nrows=1)
ds['pet'].isel(lat=100, lon=100).plot(ax=axs)
fig.savefig(os.path.join(fn_out, f"pet_randomloc_year_{year_random}.png"), dpi=300)

print("Done!")