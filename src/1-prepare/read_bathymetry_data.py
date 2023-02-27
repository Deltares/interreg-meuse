##
import os
import matplotlib.pyplot as plt
import xarray as xr
import rioxarray as rio
import hydromt
import hydromt_wflow
import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import Point, LineString
from scipy.spatial import cKDTree
import pandas as pd
import rasterio
from sympy import Point, Line
import shapely.wkt
from shapely.geometry.polygon import Polygon
from shapely.geometry import box
from shapely.geometry import MultiLineString


def linestring_to_points(feature, line):
    return {feature: line.coords}


def get_centroids_river(gpd_rivers, epsg=4326):
    """
    adapted from sobek_profiles_meuse_rhine.py

    Parameters
    ----------
    gpd_rivers : GeoDataFrame
        staticgeoms of rivers.
    epsg : int
        EPSG code. The default is 4326.

    Returns
    -------
    s : GeoSeries with centroid points of each river cell

    """

    # convert points to list of coordinates
    for i in range(len(gpd_rivers)):
        points_dict = gpd_rivers.iloc[i, :]["points"]

        points = points_dict[list(points_dict.keys())[0]]

        xys = np.array(list(zip(points.xy[0], points.xy[1])))

        if i == 0:
            total = xys.copy()
        else:
            total = np.append(total, xys, axis=0)

    x = total[:, 0]
    y = total[:, 1]

    s = gpd.GeoSeries(map(Point, zip(x, y)), )
    s.crs = epsg

    # remove duplicates
    geom_hash = [hash(tuple(geom.coords)) for geom in s.geometry]
    df_s = s.to_frame()
    df_s["hash"] = geom_hash
    df_s = df_s.drop_duplicates("hash")

    s = gpd.GeoSeries(df_s[0])
    s.crs = epsg
    return s


def ckdnearest(gdA, gdB):

    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    # gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)
    gdB_nearest = gdB.iloc[idx].reset_index(drop=True)
    gdf = pd.concat(
        [
            # gdA.reset_index(drop=True),
            gdA,
            gdB_nearest,
            pd.Series(dist, name='dist')
        ],
        axis=1)

    return gdf


def getExtrapolatedLine(p1, p2):
    """
    Creates a line extrapoled in p1->p2 direction
    """
    EXTRAPOL_RATIO = 20
    a = p1
    b = (p1[0] + EXTRAPOL_RATIO * (p2[0] - p1[0]), p1[1] + EXTRAPOL_RATIO * (p2[1] - p1[1]))
    return LineString([a, b])


# path_wflow = r'p:\11208719-interreg\wflow\b_rootzone\staticmaps.nc'
root = r'p:\11208719-interreg\wflow\b_rootzone'
path_centerlines = r'p:\11208186-spw\Data\MESU2021\MESU-CENTRELINES.shp'
# path_bathymetry = r'p:\11208186-spw\Data\BATHYMETRIE_GEOTIFF_31370_PROV_LIEGE\LIE\BATHY_50cm_ALTITUDE_DNG.tif'
# path_save = r'p:\11208186-spw\Data\BATHYMETRIE_GEOTIFF_31370_PROV_LIEGE\b_preprocess'

# the spw data has the Belgian Lambert 72 EPSG:31370 projection
# wflow = xr.open_dataset(path_wflow)
# wflow.rio.crs --> CRS.from_epsg(4326)

# bathymetry = hydromt.open_raster(path_bathymetry, chunks={'x':1000, 'y':1000} )
# bathymetry = hydromt.io.open_geodataset(path_bathymetry, bbox=[48, 4, 49, 5]) not a supported file format
# bathymetry = bathymetry.raster.reproject_like(wflow)

centerlines = gpd.read_file(path_centerlines)
# necessary because centerlines.crs gives EPSG:4326 but it is in the Belgian crs!
centerlines = centerlines.set_crs("EPSG:31370", allow_override=True)
# reproject to match the staticmaps file
centerlines = centerlines.to_crs("EPSG:4326")

# explode multilinestring to linestring
centerlines = centerlines.explode(column='geometry')
# convert centerlines to points
centerlines["points"] = centerlines.apply(lambda l: linestring_to_points(l['OBJECTID'], l['geometry']), axis=1)
# for some reason it is very slow
# get centroids of centerlines
# s = get_centroids_river(centerlines)
# s = gpd.GeoDataFrame(geometry=gpd.GeoSeries(s))
# s.rename_geometry('point_centerlines', inplace=True)
# s.to_file(r'c:\Projects\interreg\s.geojson', crs="EPSG:4326")  # seems to have worked fine
s = gpd.read_file(r'c:\Projects\interreg\s.geojson', crs="EPSG:4326")

# read wflow staticmaps
mod = hydromt_wflow.WflowModel(root=root, mode="r")
ds = mod.staticmaps
# convert xarray to tif file, based on:
# https://geopandas.org/en/stable/gallery/geopandas_rasterio_sample.html
ds["wflow_river"].raster.to_raster(r'c:\Projects\interreg\wflow_river.tif')
# read tif file
src = rasterio.open(r'c:\Projects\interreg\wflow_river.tif')

# get centroids for entire wflow area
grid = ds["wflow_river"].raster.vector_grid()
# grid.to_file(r'c:\Projects\interreg\grid.geojson', crs="EPSG:4326")
grid_centroid = grid.geometry.centroid
# convert to geodataframe
grid_centroid = gpd.GeoDataFrame(geometry=gpd.GeoSeries(grid_centroid), crs="EPSG:4326")
# sample tif file
coord_list = [(x,y) for x,y in zip(grid_centroid['geometry'].x, grid_centroid['geometry'].y)]
grid_centroid['value'] = [x for x in src.sample(coord_list)]
grid_centroid['value'] = grid_centroid['value'].astype(np.int64)
# only keep centroids over rivers
grid_centroid = grid_centroid.where(grid_centroid['value']==1).dropna()
grid_centroid.rename_geometry('point_wflow', inplace=True)
# grid_centroid.to_file(r'c:\Projects\interreg\visualization\grid_centroid.geojson', crs="EPSG:4326")

# snap point from centerlines to the nearest river centroid
snapped_gdf = ckdnearest(grid_centroid, s)

# change geometry of snapped to centerline point
snapped_gdf.set_geometry('geometry', inplace=True, crs="EPSG:4326")
# create a buffer around each selected river point from snapped_gdf
buffer = snapped_gdf.geometry.buffer(0.001)
# buffer.to_file(r'c:\Projects\interreg\visualization\buffer.geojson', crs="EPSG:4326")

# clip centerlines to buffer extent
clipped = centerlines.geometry.clip(buffer)
clipped = clipped.explode()
# convert to geodataframe
clipped = gpd.GeoDataFrame(geometry=clipped, crs="EPSG:4326")
clipped.to_file(r'c:\Projects\interreg\visualization\clipped.geojson', crs="EPSG:4326")
# clipped = clipped.rename_geometry('linegeom1')
clipped['linegeom'] = clipped.geometry

# clipped.set_geometry('linegeom', inplace=True)

# join snapped_gdf (point wflow, point centerlines) with clipped centerlines (linestring)
joined = gpd.sjoin(snapped_gdf, clipped, how="left", predicate="intersects")
joined = gpd.sjoin_nearest(snapped_gdf, clipped, how="left", max_distance=0.1, distance_col='distances')
# joined['linegeom2'].to_file(r'c:\Projects\interreg\selected_slines.geojson', crs="EPSG:4326")
# seems to be working as expected


def perpendicular(linestring):
    distance = 0.005
    left = linestring.parallel_offset(distance, 'left')
    right = linestring.parallel_offset(distance, 'right')
    return LineString([left.centroid, right.centroid])


# test_line = joined.iloc[10].linegeom
# test_perpendicular = perpendicular(test_line)
# test = gpd.GeoDataFrame({'geometry':[test_line, test_perpendicular]}, geometry='geometry', crs='EPSG:4326')
# test.to_file(r'c:\Projects\interreg\test.geojson', crs="EPSG:4326")

# TODO: apply perpendicular function to joined dataframe
# joined.set_geometry('linegeom2', inplace=True, crs="EPSG:4326")
joined['perpendicular'] = joined.apply(lambda x: perpendicular(x['linegeom2'] if len(x)>0 else ''), axis=1)
joined['perpendicular'].to_file(r'c:\Projects\interreg\perpendicular.geojson', crs="EPSG:4326")

# snapped_gdf['point_centerlines'].to_file(r'c:\Projects\interreg\selected_s.geojson', crs="EPSG:4326")
# perpendicular line
# https://stackoverflow.com/questions/21291725/determine-if-shapely-point-is-within-a-linestring-multilinestring

# snapped_gdf['line'] = snapped_gdf.apply(lambda row: getExtrapolatedLine(row['point_wflow'], row['point_centerlines']),axis=1)
# snapped_gdf['line'].to_file(r'c:\Projects\interreg\line.geojson', crs="EPSG:4326")

# rivers = mod.staticgeoms["rivers"]
# mask = grid_centroid.within(rivers)
# mask = grid_centroid.overlaps(rivers)
# mask = grid_centroid.intersects(rivers)
# grid_centroid = grid_centroid[mask]
# empty: within don't work: contains, overlaps,
# grid = gpd.sjoin(grid, rivers, how="inner", predicate="contains")
# grid.to_file(r'c:\Projects\interreg\grid_contains.geojson', crs="EPSG:4326")
# grid_centroid = grid.geometry.centroid
# grid_centroid = gpd.clip(grid_centroid, mask=rivers)
# save it and view it in QGIS
# this seems to be the way to go, although now not every grid centroid on top of a river cell is kept
# try sjoin_nearest? or apply a buffer and then join
# grid_centroid_gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(grid_centroid), crs='EPSG:4326')
# grid_centroid_gdf['geometry'] = grid_centroid_gdf.geometry.buffer(5)
# grid_centroid_rivers = gpd.sjoin(grid_centroid_gdf, rivers, how="inner", predicate="intersects")
# grid_centroid_rivers.rename_geometry('point_wflow', inplace=True)
# join.to_file(r'c:\Projects\interreg\join.geojson', crs="EPSG:4326")
# nearest_index, dst = hydromt.gis_utils.nearest(s, grid_centroid)
# KeyError: '[67288, 69225, 69922, 62447] not in index'
# selected_s = s[nearest_index]
# less than one point for each river centroid because not all river centroids are included
# selected_s = ckdnearest(join, s)
# more than one point from centerlines for each river centroid because there are also non river centroids included
# selected_s = ckdnearest(grid_centroid, s)
# selected_s.to_file(r'c:\Projects\interreg\selected_s.geojson', crs="EPSG:4326")
# create line connecting join from wflow river and selected_s from centerlines geodataframes
# snapped_gdf['line'] = snapped_gdf.apply(lambda row: LineString([row['point_wflow'], row['point_centerlines']]), axis=1)

