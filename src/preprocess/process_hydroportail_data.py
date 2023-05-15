#%%
import pandas as pd
import os
import matplotlib.pyplot as plt
import xarray as xr
import glob 

#%%

Folder_data = r"p:\11208719-interreg\data\hydroportail\a_raw"
Folder_data_processed = r"p:\11208719-interreg\data\hydroportail\c_final"

stations = {

    "chooz_diable": {
                        "folder" : "la_meuse_a_chooz",
                        "hp_id" : "B720000001",
                        "station_name" : "La Meuse a Chooz - Trou du Diable",
                        "x": 4.7825797, #wflow coords
                        "y": 50.0890491, #wflow coords
                        "wflow_id" : 1720000001, #replace B with 1
                        "rws_wflow_id": 4,
                        "Commentaire": "Cette station est située en amont de la centrale de Chooz et ne prend donc pas en compte le débit évaporé par les aéroréfrigérants."
                            },


    "chooz_graviat": {
                        "folder" : "la_meuse_a_chooz",
                        "hp_id" : "B720000002",
                        "station_name" : "La Meuse a Chooz - Ile Graviat",
                        "x": 4.7825797, #wflow coords, same as other chooz station in wflow grid
                        "y": 50.0890491, #wflow coords, same as other chooz station in wflow grid
                        "wflow_id" : 1720000002, #replace B with 1
                        "rws_wflow_id": 4,
                        "Commentaire": "Cette station est située en aval de la centrale de Chooz et prend donc en compte le débit évaporé par les aéroréfrigérants."
                            },


    "goncourt": {
                        "folder" : "la_meuse_a_goncourt",
                        "hp_id" : "B022001001",
                        "station_name" : "La Meuse a Goncourt",
                        "x": 5.6142568, #wflow coords
                        "y": 48.2410111, #wflow coords
                        "wflow_id" : 1022001001, #replace B with 1
                        "rws_wflow_id": 1011,
                        "Commentaire": "A partir du 23/03/2007, les donnees sont en TU."
                            },
}

#%%

timesteps = ["daily", "hourly"]

for timestep in timesteps:
    ds_timestep = []
    for station in stations:
        print(timestep, station)
        folder = stations[station]["folder"]
        hp_id = stations[station]["hp_id"]
        wflow_id = stations[station]["wflow_id"]
        rws_id = stations[station]["rws_wflow_id"]
        station_name = stations[station]["station_name"]
        station_x = stations[station]["x"]
        station_y = stations[station]["y"]
        remark = stations[station]["Commentaire"]

        
        fns = glob.glob(os.path.join(Folder_data, folder, timestep, f"export_hydro_series_{hp_id}*csv"))

        if timestep == "hourly":
            freq="H"
        else:
            freq="D"
        
        rng = pd.date_range(start= "1953-01-01", end = "2023-04-30 23:00:00", freq=freq)
        df_all = pd.DataFrame(index=rng, columns=["Q"])

        for fn in fns:
            print(fn)
            df = pd.read_csv(fn, sep = ";", index_col=4, parse_dates=True, skiprows=[1], decimal=',')
            df_q = df["<ResObsElaborHydro>"]
            df_all.loc[df_q.index, "Q"] = df_q/1000
        
        ds = df_all.to_xarray()
        ds = ds.rename({"index": "time"})

        #add id 
        ds = ds.assign_coords({"index":hp_id}).expand_dims(["index"])
        #add wflow id and other station related info
        ds = ds.assign_coords({"rws_id":("index",[rws_id])})
        ds = ds.assign_coords({"wflow_id":("index",[wflow_id])})
        ds = ds.assign_coords({"x":("index",[station_x])})
        ds = ds.assign_coords({"y":("index",[station_y])})
        ds = ds.assign_coords({"station_name":("index",[station_name])})

        ds = ds.astype(float)
        ds["Q"].attrs["unit"] = "m3/s"
        ds["Q"].attrs["remark"] = remark

        #swap dims
        ds = ds.swap_dims({"index":"wflow_id"})

        ds_timestep.append(ds)

    
    ds_all_timestep = xr.merge(ds_timestep)

    #write to netcdf
    ds_all_timestep.to_netcdf(os.path.join(Folder_data_processed, f"hydro_{timestep}.nc"))

#%% make csv file to add layer in wflow model 

#only goncourt and chooz trou du diable
stations_sel = [1022001001, 1720000001]

df = pd.DataFrame(index = stations_sel, columns = ["x", "y", "station_name", "rws_id", "index"])
df.index.name = "wflow_id"
df

for station in df.index:
    print(station)
    df.loc[station, "x"] = ds_all_timestep.sel(wflow_id=station)["x"].values
    df.loc[station, "y"] = ds_all_timestep.sel(wflow_id=station)["y"].values
    df.loc[station, "rws_id"] = ds_all_timestep.sel(wflow_id=station)["rws_id"].values
    df.loc[station, "index"] = ds_all_timestep.sel(wflow_id=station)["index"].values
    df.loc[station, "station_name"] = ds_all_timestep.sel(wflow_id=station)["station_name"].values
    
df.to_csv(os.path.join(Folder_data_processed, "hydroportail_stations.csv"))    


ds_all_timestep.wflow_id.values

ds_all_timestep.sel(wflow_id = 1022001001) 


#%% checks !
#previously downloaded
# dd = xr.open_dataset(r"p:\11208719-interreg\data\observed_streamflow_grade\FR-Hydro-hourly-2005_2022.nc")

# plt.figure(); 
# ds_all_timestep.Q.sel(index = ["B720000001", "B720000002"]).plot(hue="index")
# dd["Q"].sel(wflow_id=4).plot()

# plt.figure(); ds_all_timestep.Q.sel(index = "B022001001").plot()
# dd["Q"].sel(wflow_id=1011).plot()

# # plt.figure(); df_all["Q"].plot()

