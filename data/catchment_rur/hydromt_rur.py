import numpy as np
import hydromt
import xarray as xr

def set_vectorfile(mod, path, param_wflow, param_source, crs, scale_factor=1, new_name=None):
    print(f"Running set_vectorfile for {param_wflow}")
    basin_geom = mod.staticgeoms["basins"]
    gdf = hydromt.io.open_vector(path, geom=basin_geom, crs=crs)
    try:
        ds = mod.staticmaps[param_wflow]
    except:
        return print("Unable to find Wflow model")
    ds_vectorfile = ds.raster.rasterize(gdf, col_name=param_source, nodata=0)
    ds = xr.where(ds_vectorfile > 0, ds_vectorfile*scale_factor, ds)
    mapname = new_name if new_name else param_wflow
    mod.set_staticmaps(ds, mapname)
    print(f"New map for {param_wflow} added")
    return

def set_river(mod, mapname, river_path, river_col, crs, replace=None):
    gdf = hydromt.io.open_vector(river_path, crs=crs)
    ds = mod.staticmaps[mapname]
    if replace:
        ds = xr.where(ds > 0, replace, np.nan)
    ds_new = ds.raster.rasterize(gdf, col_name=river_col, nodata=np.nan, all_touched=True)
    ds_merged = xr.where(ds_new > 0, ds_new, ds)
    ds_merged = xr.where(ds_merged > 0, ds_merged, np.nan)
    mod.set_staticmaps(ds_merged, mapname)
    return

def set_usinggeom(mod, path, param_wflow, default_value, index_col, crs, *args):
    print(f"Running set_usinggeom for {param_wflow} for {len(args)} regions")
    ds = mod.staticmaps["wflow_dem"]
    ds = xr.where(ds > 0, default_value, 0)
    basin_geom = mod.staticgeoms["basins"]
    gdf = hydromt.io.open_vector(path, geom=basin_geom, crs=crs)
    for region in args:
        clip_geom = gdf[gdf[index_col] == region[index_col]]
        ds_masked = ds.raster.geometry_mask(clip_geom, all_touched=True)
        ds_masked = xr.where(ds_masked > 0, region[param_wflow], 0)
        ds = xr.where(ds_masked > 0, ds_masked, ds)
    mod.set_staticmaps(ds, param_wflow)
    print(f"New map for {param_wflow} added")
    return