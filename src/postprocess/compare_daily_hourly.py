#%%
import pandas as pd
import os
import numpy as np 
import xarray as xr 
import matplotlib.pyplot as plt

case_dir = r"p:\11208719-interreg\wflow\d_manualcalib"

run_hourly_case = "run_manualcalib"
run_daily_case = "run_manualcalib_daily_genre"
run_daily_eobs = "run_manualcalib_daily_eobs24"

df_daily = pd.read_csv(os.path.join(case_dir, run_daily_case, "output.csv"), index_col=0, parse_dates=True, header=0)
df_daily_eobs = pd.read_csv(os.path.join(case_dir, run_daily_eobs, "output.csv"), index_col=0, parse_dates=True, header=0)
df_hourly = pd.read_csv(os.path.join(case_dir, run_hourly_case, "output.csv"), index_col=0, parse_dates=True, header=0)

#%%
catch = 10
fig, (ax1,ax2) = plt.subplots(2,1, sharex=True)
df_hourly[f"Q_{catch}"].resample("D").mean().plot(ax=ax1, label="genre hourly")
df_daily[f"Q_{catch}"].plot(ax=ax1, label="genre daily")
df_daily_eobs[f"Q_{catch}"].plot(ax=ax1, label="eobs")
ax1.legend()

df_hourly[f"P_{catch}"].resample("D", ).sum().plot(ax=ax2) #label="right", closed="right"
df_daily[f"P_{catch}"].plot(ax=ax2)



fig, (ax1,ax2) = plt.subplots(2,1, sharex=True)
var = "EI"
df_hourly[f"{var}_{catch}"].resample("D").sum().plot(ax=ax1)
df_daily[f"{var}_{catch}"].plot(ax=ax1)
df_daily_eobs[f"{var}_{catch}"].plot(ax=ax1, label="eobs")
ax1.legend()

var = "ustoredepth"
df_hourly[f"{var}_{catch}"].resample("D").mean().plot(ax=ax2)
df_daily[f"{var}_{catch}"].plot(ax=ax2)
df_daily_eobs[f"{var}_{catch}"].plot(ax=ax2, label="eobs")
ax1.legend()

#%%

df_hourly[f"P_{catch}"].resample("D", ).sum().loc["2007-01-16":"2007-01-19"].sum()
df_daily[f"P_{catch}"].loc["2007-01-16":"2007-01-19"].sum()


plt.figure(); df_hourly[f"P_{catch}"].loc["2007-01-16":"2007-01-19"].plot()