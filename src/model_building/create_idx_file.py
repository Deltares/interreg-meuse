import hydromt
import xarray as xr
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import os


def create_idx_file(forcing_fn, staticmaps_fn, outfile_fn, plotting=False):
    # Load the raw dataset
    raw_data = xr.open_dataset(forcing_fn, chunks={"time": 10})

    # Load the target dataset
    target_data = xr.open_dataset(staticmaps_fn)

    orig_lat = raw_data.lat.values
    orig_lon = raw_data.lon.values

    xdim = target_data.raster.x_dim
    ydim = target_data.raster.y_dim

    target_lon, target_lat = np.meshgrid(
        target_data.raster.xcoords, target_data.raster.ycoords
    )

    # flatten the original latitude and longitude arrays
    orig_lat_flat = orig_lat.flatten()
    orig_lon_flat = orig_lon.flatten()

    # stack the flattened arrays horizontally
    orig_lat_lon = np.column_stack((orig_lat_flat, orig_lon_flat))

    # stack the target latitude and longitude arrays horizontally
    target_lat_lon = np.column_stack((target_lat.flatten(), target_lon.flatten()))

    # calculate the distance between each target point and all the original points
    distances = cdist(target_lat_lon, orig_lat_lon)

    # get the index of the minimum distance for each target point
    min_idx = np.argmin(distances, axis=1)

    # convert the flat index to 2D indices
    min_idx_2d = np.unravel_index(min_idx, orig_lat.shape)

    # reshape the indices to match the target shape
    min_idx_2d = np.column_stack(min_idx_2d).reshape(target_lat.shape + (2,))

    if plotting:
        plt.figure()
        plt.imshow(min_idx_2d[:, :, 0])
        plt.figure()
        plt.imshow(min_idx_2d[:, :, 1])

    #Changed based on discussion with Joos - Pyhton vs Julia indexing --> difference of 1
    lat_idx = min_idx_2d[:, :, 0] + 1
    lon_idx = min_idx_2d[:, :, 1] + 1

    target_data["lat_idx"] = ((ydim, xdim), lat_idx)
    target_data["lon_idx"] = ((ydim, xdim), lon_idx)

    output = target_data[["lat_idx", "lon_idx", "wflow_dem"]]
    output.to_netcdf(outfile_fn)
    return output

# if __name__ == "__main__":
#     forcing_fn = R"p:\11209265-grade2023\climate_scenarios\scenarios_23\1_input_240\fromKNMI_wflow\RACMO23_coarse_voorWflow\conversionDone\merged\RACMO23_RESAMPLES_all_resample_control_2100Hn_definitive_biascorrected.nc"
#     staticmaps_fn = R"p:\11209265-grade2023\climate_scenarios\scenarios_23\3_Wflow\Meuse\wflow_202305\staticmaps\staticmaps_routing_cal_11.nc"
#     outfile_fn = R"p:\11209265-grade2023\climate_scenarios\scenarios_23\3_Wflow\Meuse\wflow_202305\staticmaps\forcing_idx.nc"

#     idx_data = create_idx_file(
#         forcing_fn=forcing_fn, staticmaps_fn=staticmaps_fn, outfile_fn=outfile_fn
#     )

#%% extract params and file locations from snakemake
outfile_fn = str(snakemake.output)
forcing_fn = snakemake.input.fn_in

#extracting params
model = snakemake.params.model
fn = snakemake.params.fn_wflow

staticmaps_fn = os.path.abspath(os.path.join(fn, model, 'staticmaps.nc'))

print("------- Checking what we got ------")
print("Model is: ", model)
print("Model location is: ", fn)
print("Forcing is: ", forcing_fn)
print("Output file:", outfile_fn)

#%%
idx_data = create_idx_file(forcing_fn=forcing_fn, staticmaps_fn=staticmaps_fn, outfile_fn=outfile_fn)

print("------- idx file created ------")