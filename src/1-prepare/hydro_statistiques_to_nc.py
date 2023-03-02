import xarray as xr
import pandas as pd
import os
import glob

folder_p = r'p:\11208719-interreg\data\hydroportail\a_raw'
folder_preprocessed = r'p:\11208719-interreg\data\hydroportail\b_preprocessed'
folder_final = r'p:\11208719-interreg\data\hydroportail\c_final'

daily = 'QJ-X_(CRUCAL)'
# this is wrong
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
            da.to_netcdf(os.path.join(folder_preprocessed, station[i] + '.nc'))


# file1 = os.path.join(folder_p, 'la_meuse_a_chooz', timestep, 'QJ-X-(CRUCAL)_Gumbel_B7200000_01-01-1953_06-02-2023_1_non-glissant_X_pre-valide-et-valide_ResultatsAjustement.csv')
# file2 = os.path.join(folder_p, 'la_meuse_a_goncourt', daily, 'QJ-X-(CRUCAL)_Gumbel_B022001001_15-09-1971_06-02-2023_1_non-glissant_X_pre-valide-et-valide_ResultatsAjustement.csv')
# l'autre fichier contient le mot Echantillon

# df1 = pd.read_csv(file1, index_col='Période de retour')
# df2 = pd.read_csv(file2, index_col='Période de retour')
#
# df1.index.rename('T', inplace=True)
# df2.index.rename('T', inplace=True)
#
# df1.rename(columns={'Valeur ajustée (en m³/s)': 'valeur', 'Intervalle de confiance bas (en m³/s)': 'int_conf_bas',
#                     'Intervalle de confiance haut (en m³/s)': 'int_conf_haut'}, inplace=True)
# df2.rename(columns={'Valeur ajustée (en m³/s)': 'valeur', 'Intervalle de confiance bas (en m³/s)': 'int_conf_bas',
#                     'Intervalle de confiance haut (en m³/s)': 'int_conf_haut'}, inplace=True)
#
# df1.drop('Fréquence au non dépassement', axis=1, inplace=True)
# df2.drop('Fréquence au non dépassement', axis=1, inplace=True)
#
# da1 = df1.to_xarray()
# da2 = df2.to_xarray()
#
# da1 = da1.assign(name='la_meuse_a_chooz')
# da2 = da1.assign(name='la_meuse_a_goncourt')
#
# lat1 = 7000411
# lon1 = 827688
#
# lat2 = 6796620
# lon2 = 894048
#
# da1 = da1.assign_coords(coords={"lat": lat1})
# da1 = da1.assign_coords(coords={"lon": lon1})
#
# da2 = da2.assign_coords(coords={"lat": lat2})
# da2 = da2.assign_coords(coords={"lon": lon2})

# ds = xr.concat([da1, da2], dim='station')
ds['T'].attrs = {"units": "years", "long_name": "return period", "hydrological_year": "01/09-31/08"}
ds['valeur'].attrs = {"units": "m³/s", "description": "valeur ajustée avec la loi de Gumbel estimée par la méthode L-moments"}
ds['int_conf_haut'].attrs = {"units": "m³/s", "description": "intervalle de confiance haut à 95% quantifiée par la méthode Bootstrap paramétrique"}
ds['int_conf_bas'].attrs = {"units": "m³/s", "description": "intervalle de confiance bas à 95% quantifiée par la méthode Bootstrap paramétrique"}

# ds.to_netcdf(os.path.join(folder_save, timestep_name, 'statistiques_' + timestep_name + '.nc'))