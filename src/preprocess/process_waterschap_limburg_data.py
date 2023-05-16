import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

folder_data = r"p:\11208719-interreg\data\waterschap_limburg"

#path geul data
q_geul_fn = os.path.join(folder_data, "a_raw", "TUD_Geul_afvoeren.csv")
#read data
q_geul = pd.read_csv(q_geul_fn, dayfirst=True, parse_dates=True, header=0, index_col=0, skiprows=[1])
#replace -999
q_geul = q_geul.replace(-999, np.nan)
#index to datetime
q_geul.index = pd.to_datetime(q_geul.index)
#resample
q_geul_h = q_geul.resample("H").mean()
q_geul_d = q_geul.resample("D").mean()
#plot 
plt.figure(); q_geul_d.plot()
plt.figure(); q_geul_h.plot()
#write to file
q_geul_h.to_csv(os.path.join(folder_data, "b_preprocessed", "hydro_geul_hourly.csv"))
q_geul_d.to_csv(os.path.join(folder_data, "b_preprocessed", "hydro_geul_daily.csv"))

(q_geul_h["12.Q.46"] + q_geul_h["12.Q.31"]).plot()


# data geul angela
q_geul_2 = pd.read_csv(os.path.join(folder_data, "a_raw", "Q_Geul_project", "waterschap_limburg", "geul_afvoeren.csv"), dayfirst=True, parse_dates=True, header=0, index_col=0, skiprows=[1])
q_geul_2 = q_geul_2.replace(-999, np.nan)
q_geul_2.index = pd.to_datetime(q_geul_2.index)
q_geul_2_h = q_geul_2.resample("H").mean()
q_geul_2_d = q_geul_2.resample("D").mean()

#check if data is the same - angela and received from klaasjan
fig, ax = plt.subplots()
q_geul_2_h["13.Q.34"].plot()
q_geul_h["13.Q.34"].plot(linestyle="--")

fig, ax = plt.subplots()
q_geul_2_h["10.Q.30"].plot()
q_geul_h["10.Q.30"].plot(linestyle="--")


#path Roer data 
q_roer_fn = os.path.join(folder_data, "a_raw", "TUD_Roer_afvoeren.csv")
#read data
q_roer = pd.read_csv(q_roer_fn, dayfirst=True, parse_dates=True, header=0, index_col=0, skiprows=[1])
#replace -999
q_roer = q_roer.replace(-999, np.nan)
#index to datetime
q_roer.index = pd.to_datetime(q_roer.index)
#resample
q_roer_h = q_roer.resample("H").mean()
q_roer_d = q_roer.resample("D").mean()
#plot 
plt.figure(); q_roer_d.plot()
plt.figure(); q_roer_h.plot()
#write to file
q_roer_h.to_csv(os.path.join(folder_data, "b_preprocessed", "hydro_roer_hourly.csv"))
q_roer_d.to_csv(os.path.join(folder_data, "b_preprocessed", "hydro_roer_daily.csv"))


#%%make csv to create gauges in wflow. 
locations = pd.read_csv(os.path.join(folder_data, "a_raw", "WL_loc_Q_stations.csv"), sep=";", index_col=0)
df = pd.DataFrame(index = list(q_roer_h.columns.values) + list(q_geul_h.columns.values), columns = ["x", "y", "station_name", "wflow_id"])
df.index.name = "index"

for index in df.index:
    print(index)
    df.loc[index, "wflow_id"] = index.replace(".Q.", "")
    if index in locations.index:
        df.loc[index, "station_name"] = locations.loc[index, "Name"]
        df.loc[index, "x"] = locations.loc[index, "x"]
        df.loc[index, "y"] = locations.loc[index, "y"]

#copy original x and y coord 
df["x_orig"] = df["x"]
df["y_orig"] = df["y"]

# a bit of manual cleaning up and moving
# Selzerbeek Partij + Selzerbeek Molentak ["12.Q.46"] ["12.Q.31"] should be summed
# move coordinate Jeker at Nekum because does not match wflow river 
# remove St. Gillisstraat	?? not on a river 

#make some manual changes to make sure points are on wflow river. (based on inspection in qgis)
#jeker
df.loc["16.Q.42", "x" ] = 5.6705
df.loc["16.Q.42", "y" ] = 50.7901

#gulp
df.loc["13.Q.34", "x" ] = 5.88391
df.loc["13.Q.34", "y" ] = 50.81439

#hommerich 
df.loc["10.Q.30", "x" ] = 5.92018
df.loc["10.Q.30", "y" ] = 50.79991

#remove molentak and St. Gillisstraat 
df = df.drop(index=["12.Q.46", "10.Q.35"])

#add Meerssen and Schin op Geul (available in data from Angela)
df.loc["10.Q.36"] = [
                    5.7258,
                    50.89121,
                    "Geul Meerssen", 
                   1036, 
                    5.7258,
                    50.89121,
                   ]

#add Schin op Geul (available in data from Angela)
df.loc["10.Q.63"] = [
                    5.86915,
                    50.85476,
                    "Geul Schin op Geul", 
                   1063, 
                    5.86915,
                    50.85476,
                   ]

#add Hambeek even though no data for it now 
df.loc["2.Q.6"] = [locations.loc["2.Q.6", "x"],
                   locations.loc["2.Q.6", "y"],
                   locations.loc["2.Q.6", "Name"],
                   26, 
                   locations.loc["2.Q.6", "x"],
                   locations.loc["2.Q.6", "y"],
                   ]


#make wflow_id the index
df["index"] = df.index
df.index = df.wflow_id
df = df.drop(columns="wflow_id")

#indicate that selzerbeek partij and molentak are summed 
df.loc["1231", "station_name"]  = "Selzerbeek, Partij + Selzerbeek, Molentak"
df.loc["1231", "index"]  = "12.Q.31 + 12.Q.46"

df.to_csv(os.path.join(folder_data, "c_final", "waterschap_limburg_stations.csv"))






#%% convert to dataset

# add meerssen en schin op geul from data Angela

timesteps = ["D", "H"]

for timestep in timesteps:
    print (timestep)
    rng = pd.date_range("1969-01-01", "2021-12-31", freq = timestep)

    ds = xr.Dataset(
        data_vars = dict(
        Q=(["wflow_id", "time"], np.zeros((len(df.index), len(rng))) *np.nan ),
        ),

        coords = dict(
        wflow_id = df.index.astype(int),
        index = ("wflow_id", df["index"].astype(str)),
        name = ("wflow_id", df.station_name.astype(str)),
        x = ("wflow_id", df.x.astype(float)),
        y = ("wflow_id", df.y.astype(float)),
        x_orig = ("wflow_id", df.x_orig.astype(float)),
        y_orig = ("wflow_id", df.y_orig.astype(float)),
        time = rng,
        )
    )

    ds = ds.swap_dims({"wflow_id":"index"})

    if timestep == "H":
        data = pd.concat([q_roer_h, q_geul_h, q_geul_2_h[["10.Q.36", "10.Q.63"]]], axis = 1)
    else:
        data = pd.concat([q_roer_d, q_geul_d, q_geul_2_d[["10.Q.36", "10.Q.63"]]], axis = 1)
    
    for index in ds.index.values: #data.columns:
        print(index)

        if index == '12.Q.31 + 12.Q.46': #selzerbeek sum 
            print("true")
            ds["Q"].loc[dict(index = index, time = data.index)] = data["12.Q.31"] + data["12.Q.46"]

        elif index in data.columns:
            ds["Q"].loc[dict(index = index, time = data.index)] = data[index]

    #reswap dims
    ds = ds.swap_dims({"index": "wflow_id"})

    ds.to_netcdf(os.path.join(folder_data, "c_final", f"hydro_{timestep}_wl.nc"))


#%% plots and checks 

# for station in ds.wflow_id:
#     plt.figure()
#     ds["Q"].sel(wflow_id = station).plot()

