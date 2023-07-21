import xarray as xr
import matplotlib.pyplot as plt

st_a = xr.open_dataset(r"p:\11208719-interreg\wflow\a_floodplain1d\staticmaps.nc")
st_j = xr.open_dataset(r"p:\11208719-interreg\wflow\j_waterschaplimburg\staticmaps.nc")
st_k = xr.open_dataset(r"p:\11208719-interreg\wflow\k_snakecal\staticmaps.nc")
st_d = xr.open_dataset(r"p:\11208719-interreg\wflow\d_manualcalib\staticmaps.nc")

st_grade = xr.open_dataset(r"p:\11209265-grade2023\wflow\wflow_meuse_julia\wflow_meuse_202303\staticmaps_routing_cal_11.nc")

plt.figure(); st_a["ksathorfrac_sub"].plot(vmin=50, vmax=300)
plt.figure(); st_j["ksathorfrac_sub"].plot(vmin=50, vmax=300)
plt.figure(); st_d["ksathorfrac_sub"].plot(vmin=50, vmax=300)
plt.figure(); st_grade["ksathorfrac_sub"].plot(vmin=50, vmax=300)

plt.figure(); st_a["ksathorfrac_sub_ardennes"].plot(vmin=50, vmax=300)
plt.figure(); st_j["ksathorfrac_sub_ardennes"].plot(vmin=50, vmax=300)
plt.figure(); st_d["ksathorfrac_sub_ardennes"].plot(vmin=50, vmax=300)
plt.figure(); st_grade["ksathorfrac_sub_ardennes"].plot(vmin=50, vmax=300)

fig, (ax1,ax2,ax3) = plt.subplots(1,3)
st_j["SoilThickness_manual_cal"].plot(ax=ax1, vmin = 100, vmax=4000)
st_k["SoilThickness_cal"].plot(ax=ax2, vmin = 100, vmax=4000)
(st_j["SoilThickness_manual_cal"] -st_k["SoilThickness_cal"]).plot(ax=ax3)
# plt.figure(); (st_j["SoilThickness_manual_cal"] -st_k["SoilThickness_cal"]).plot()
#over het algemeen lagere soilthickness in k (in de ardennen)

fig, (ax1,ax2,ax3) = plt.subplots(1,3)
st_j["ksathorfrac_sub"].plot(ax=ax1, vmin = 100, vmax = 1000)
st_k["ksathorfrac_sub_cal"].plot(ax=ax2, vmin = 100, vmax = 1000)
(st_j["ksathorfrac_sub"] -st_k["ksathorfrac_sub_cal"]).plot(ax=ax3, vmin=-2000,vmax=2000, cmap = "RdBu_r")
# plt.figure(); (st_j["ksathorfrac_sub"] -st_k["ksathorfrac_sub_cal"]).plot(vmin=-2000,vmax=2000, cmap = "RdBu")
#over het algemeen hogere ksathorfrac in k (in de ardennen)

fig, (ax1,ax2,ax3) = plt.subplots(1,3)
st_j["Swood"].plot(ax=ax1, vmin = 0, vmax = 2.5)
st_k["Swood_cal"].plot(ax=ax2, vmin = 0, vmax = 2.5)
(st_j["Swood"] -st_k["Swood_cal"]).plot(ax=ax3,)# vmin=-2000,vmax=2000, cmap = "RdBu_r")
#over het algemeen hogere Swood in k

fig, (ax1,ax2,ax3) = plt.subplots(1,3)
st_j["N"].plot(ax=ax1, vmin = 0, vmax = 0.9)
st_k["N_cal"].plot(ax=ax2, vmin = 0, vmax = 0.9)
(st_j["N"] -st_k["N_cal"]).plot(ax=ax3,)# vmin=-2000,vmax=2000, cmap = "RdBu_r")
#over het algemeen hogere n in k

fig, (ax1,ax2,ax3) = plt.subplots(1,3)
st_j["RootingDepth_obs_20"].plot(ax=ax1, vmin = 200, vmax = 1500)
st_k["RootingDepth_obs_20_cal"].plot(ax=ax2, vmin = 200, vmax = 1500)
(st_j["RootingDepth_obs_20"] -st_k["RootingDepth_obs_20_cal"]).plot(ax=ax3,)# vmin=-2000,vmax=2000, cmap = "RdBu_r")
#over het algemeen lagere rooting depth in k

fig, (ax1,ax2,ax3) = plt.subplots(1,3)
st_j["MaxLeakage_manual_cal"].plot(ax=ax1, vmin = 0, vmax = 1)
st_k["MaxLeakage_cal"].plot(ax=ax2, vmin = 0, vmax = 1)
(st_j["MaxLeakage_manual_cal"] -st_k["MaxLeakage_cal"]).plot(ax=ax3,)# vmin=-2000,vmax=2000, cmap = "RdBu_r")
#over het algemeen hogere leakage in k!! 