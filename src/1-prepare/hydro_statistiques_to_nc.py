import numpy as np
import xarray as xr
import pandas as pd
import os
import glob

folder_p = r'p:\11208719-interreg\data\hydroportail\a_raw'
folder_preprocessed = r'p:\11208719-interreg\data\hydroportail\b_preprocessed'
folder_final = r'p:\11208719-interreg\data\hydroportail\c_final'

daily = 'QJ-X_(CRUCAL)'
hourly = 'Q-X_(CRUCAL)'

timestep = daily  # or hourly

if timestep == daily:
    timestep_name = "daily"
else:
    timestep_name = "hourly"

# add years and hydrological years
# add x y coordinates

station = ['la_meuse_a_chooz', 'la_meuse_a_goncourt']
lat = [7000411, 6796620]
lon = [827688, 894048]

for i in range(len(station)):
    for file in os.listdir(os.path.join(folder_p, station[i], timestep)):
        print(file)
        if 'ResultatsAjustement' in file:

            # create a dataframe
            df = pd.read_csv(os.path.join(folder_p, station[i], timestep, file), index_col='Période de retour')
            df.index.rename('T', inplace=True)
            df.rename(
                columns={'Valeur ajustée (en m³/s)': 'valeur', 'Intervalle de confiance bas (en m³/s)': 'int_conf_bas',
                         'Intervalle de confiance haut (en m³/s)': 'int_conf_haut'}, inplace=True)
            df.drop('Fréquence au non dépassement', axis=1, inplace=True)

            # create a dataarray
            da = df.to_xarray()
            da = da.assign(name=i)
            da = da.assign_coords(coords={"lat": lat[i]})
            da = da.assign_coords(coords={"lon": lon[i]})
            da.to_netcdf(os.path.join(folder_preprocessed, timestep_name, station[i] + '.nc'))

ds = xr.open_mfdataset(os.path.join(folder_preprocessed, timestep_name, '*.nc'), concat_dim='station', combine="nested")
ds['T'].attrs = {"units": "years", "long_name": "return period", "hydrological_year": "01/09-31/08"}
ds['valeur'].attrs = {"units": "m³/s", "description": "valeur ajustée avec la loi de Gumbel estimée par la méthode L-moments"}
ds['int_conf_haut'].attrs = {"units": "m³/s", "description": "intervalle de confiance haut à 95% quantifiée par la méthode Bootstrap paramétrique"}
ds['int_conf_bas'].attrs = {"units": "m³/s", "description": "intervalle de confiance bas à 95% quantifiée par la méthode Bootstrap paramétrique"}
ds.to_netcdf(os.path.join(folder_final, 'hydro_statistiques_' + timestep_name + '.nc'), encoding={'_FillValue': np.nan})