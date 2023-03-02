import numpy as np
import xarray as xr
import pandas as pd
import os

# manually create the data as the files are in pdf
# add hydrological years!
path_save = r'p:\11208719-interreg\data\spw\statistiques\b_preprocessed'

# extracteur
MA = "maxima annuels"
POT = "peak over threshold"
T = [25, 50, 75, 100]

# add station name and x y coordinates
# Amblève_Martinrive_LN_MA_AH.pdf
data = {'T': [25, 50, 75, 100], 'valeur': [341, 387, 414, 433], 'int_conf_haut': [409, 472, 509, 537],
        'int_conf_bas': [273, 302, 318, 329]}
station = "ambleve_martinrive"
x = 240021
y = 130556
years = np.arange(1974, 2017 + 1, 1)
extracteur = MA

ds = xr.Dataset(
        data_vars=dict(
                valeur=(['x', 'y', 'T'], np.array([[[341, 387, 414, 433]]])),
                int_conf_haut=(['x', 'y', 'T'], np.array([[[409, 472, 509, 537]]])),
                int_conf_bas=(['x', 'y', 'T'], np.array([[[273, 302, 318, 329]]])),
                        ),
        coords=dict(
                lon=(['x', 'y'], np.reshape(x, (1, 1))),
                lat=(['x', 'y'], np.reshape(y, (1, 1))),
                name=station,
                T=T,
        ),
)

ds.to_netcdf(os.path.join(path_save, station + '.nc'))

# Lesse_Gendron_LN_MA_AH.pdf
data = {'T': [25, 50, 75, 100], 'valeur': [381, 437, 471, 495], 'int_conf_haut': [470, 551, 600, 636],
        'int_conf_bas': [291, 323, 341, 353]}
station = "lesse_gendron"
x = 192424
y = 100222
years = np.arange(1974, 2011 + 1, 1)
extracteur = MA

ds = xr.Dataset(
        data_vars=dict(
                valeur=(['x', 'y', 'T'], np.array([[[381, 437, 471, 495]]])),
                int_conf_haut=(['x', 'y', 'T'], np.array([[[470, 551, 600, 636]]])),
                int_conf_bas=(['x', 'y', 'T'], np.array([[[291, 323, 341, 353]]])),
                        ),
        coords=dict(
                lon=(['x', 'y'], np.reshape(x, (1, 1))),
                lat=(['x', 'y'], np.reshape(y, (1, 1))),
                name=station,
                T=T,
        ),
)

ds.to_netcdf(os.path.join(path_save, station + '.nc'))

# Ourthe_Tabreux_LN2_MA_AH.pdf
data = {'T': [25, 50, 75, 100], 'valeur': [372, 424, 454, 476], 'int_conf_haut': [446, 516, 558, 589],
        'int_conf_bas': [299, 331, 350, 362]}
station = "ourthe_tabreux"
x = 232853
y = 125886
years = np.arange(1974, 2020 + 1, 1)
extracteur = MA

ds = xr.Dataset(
        data_vars=dict(
                valeur=(['x', 'y', 'T'], np.array([[[372, 424, 454, 476]]])),
                int_conf_haut=(['x', 'y', 'T'], np.array([[[446, 516, 558, 589]]])),
                int_conf_bas=(['x', 'y', 'T'], np.array([[[299, 331, 350, 362]]])),
                        ),
        coords=dict(
                lon=(['x', 'y'], np.reshape(x, (1, 1))),
                lat=(['x', 'y'], np.reshape(y, (1, 1))),
                name=station,
                T=T,
        ),
)

ds.to_netcdf(os.path.join(path_save, station + '.nc'))

# Sambre_Floriffoux_WeibullMM3_POT.pdf
data = {'T': [25, 50, 75, 100], 'valeur': [408, 441, 458, 473], 'int_conf_haut': [454, 493, 515, 534],
        'int_conf_bas': [362, 389, 402, 414]}
station = "sambre_floriffoux"
x = 178824
y = 126390
years = np.arange(1990, 2004 + 1, 1)
extracteur = POT

ds = xr.Dataset(
        data_vars=dict(
                valeur=(['x', 'y', 'T'], np.array([[[408, 441, 458, 473]]])),
                int_conf_haut=(['x', 'y', 'T'], np.array([[[454, 493, 515, 534]]])),
                int_conf_bas=(['x', 'y', 'T'], np.array([[[362, 389, 402, 414]]])),
                        ),
        coords=dict(
                lon=(['x', 'y'], np.reshape(x, (1, 1))),
                lat=(['x', 'y'], np.reshape(y, (1, 1))),
                name=station,
                T=T,
        ),
)

ds.to_netcdf(os.path.join(path_save, station + '.nc'))

# Semois_Membre_Gamma_MA_AH.pdf
data = {'T': [25, 50, 75, 100], 'valeur': [484, 539, 570, 591], 'int_conf_haut': [574, 645, 686, 714],
        'int_conf_bas': [394, 432, 453, 468]}
station = "semois_membre"
x = 188327
y = 61505
years = np.arange(1974, 2011 + 1, 1)
extracteur = MA

ds = xr.Dataset(
        data_vars=dict(
                valeur=(['x', 'y', 'T'], np.array([[[484, 539, 570, 591]]])),
                int_conf_haut=(['x', 'y', 'T'], np.array([[[574, 645, 686, 714]]])),
                int_conf_bas=(['x', 'y', 'T'], np.array([[[394, 432, 453, 468]]])),
                        ),
        coords=dict(
                lon=(['x', 'y'], np.reshape(x, (1, 1))),
                lat=(['x', 'y'], np.reshape(y, (1, 1))),
                name=station,
                T=T,
        ),
)

ds.to_netcdf(os.path.join(path_save, station + '.nc'))

# Vesdre_Chaudfontaine_WeibullMV_AH_2.pdf
data = {'T': [25, 50, 75, 100], 'valeur': [226, 241, 250, 255], 'int_conf_haut': [np.nan, np.nan, np.nan, np.nan],
        'int_conf_bas': [np.nan, np.nan, np.nan, np.nan]}
station = "vesdre_chaudfontaine"
x = 240981
y = 142875
years = np.arange(1974, 2005 + 1, 1)
extracteur = ""

# This works!
ds = xr.Dataset(
        data_vars=dict(
                valeur=(['x', 'y', 'T'], np.array([[[226, 241, 250, 255]]])),
                int_conf_haut=(['x', 'y', 'T'], np.array([[[np.nan, np.nan, np.nan, np.nan]]])),
                int_conf_bas=(['x', 'y', 'T'], np.array([[[np.nan, np.nan, np.nan, np.nan]]])),
                        ),
        coords=dict(
                lon=(['x', 'y'], np.reshape(x, (1, 1))),
                lat=(['x', 'y'], np.reshape(y, (1, 1))),
                name=station,
                T=T,
        ),
)

ds.to_netcdf(os.path.join(path_save, station + '.nc'))

# ---------------------------------------------------------------------------------------------------------------------
# ds created manually from 6 stations
# ---------------------------------------------------------------------------------------------------------------------
T = [25, 50, 75, 100]
x = [240021, 192424, 232853, 178824, 188327, 240981]
y = [130556, 100222, 125886, 126390, 61505, 142875]
station = ["ambleve_martinrive", "lesse_gendron", "ourthe_tabreux", "sambre_floriffoux", "semois_membre", "vesdre_chaudfontaine"]
valeur = [[[341, 387, 414, 433], [381, 437, 471, 495], [372, 424, 454, 476], [408, 441, 458, 473], [484, 539, 570, 591], [226, 241, 250, 255]]]
int_conf_haut = [[[409, 472, 509, 537], [470, 551, 600, 636], [446, 516, 558, 589], [454, 493, 515, 534], [574, 645, 686, 714], [np.nan, np.nan, np.nan, np.nan]]]
int_conf_bas =[[[273, 302, 318, 329], [291, 323, 341, 353], [299, 331, 350, 362], [362, 389, 402, 414], [394, 432, 453, 468], [np.nan, np.nan, np.nan, np.nan]]]

ds = xr.Dataset(
        data_vars=dict(
                valeur=(['x', 'y', 'T'], valeur),
                int_conf_haut=(['x', 'y', 'T'], int_conf_haut),
                int_conf_bas=(['x', 'y', 'T'], int_conf_bas),
                name=(['x', 'y'], np.reshape(station, (1, -1))),
        ),
        coords=dict(
                lon=(['x', 'y'], np.reshape(x, (1, -1))),
                lat=(['x', 'y'], np.reshape(y, (1, -1))),
                T=T,
        ),
        attrs=dict(
                title="Estimation des debits extremes",
                institution='Service public de la Wallonie',
                crs='Coordonnées Lambert'
        )
)

# add attributes
ds['T'].attrs['units'] = 'years'
ds['T'].attrs['long_name'] = 'return period'
ds['T'].attrs['hydrological_year'] = '01/10-30/09'

ds['int_conf_haut'].attrs['long_name'] = 'intervalle de confiance haut à 95%'
ds['int_conf_bas'].attrs['long_name'] = 'intervalle de confiance bas à 95%'


ds = ds.squeeze()
ds = ds.rename({"y": "station"})

ds.to_netcdf(os.path.join(r'p:\11208719-interreg\data\spw\statistiques\c_final', 'spw_statistiques.nc'))
