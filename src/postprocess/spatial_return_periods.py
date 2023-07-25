#%%
import pandas as pd
import matplotlib.pyplot as plt
from hydromt.stats import extremes
import os
import xarray as xr
import numpy as np
import xclim
from xclim.ensembles import create_ensemble
import glob
import time

#%% Functions
def spatial_T_analysis(ds_sub, Ts):
    qs = [1-1/t for t in Ts]
    T_gumb = xclim.indices.stats.fa(ds_sub.chunk(dict(time=-1)), t=Ts, dist="gumbel_r", mode="max")
    T_gev = xclim.indices.stats.fa(ds_sub.chunk(dict(time=-1)), t=Ts, dist="genextreme", mode="max")
    T_emp = ds_sub.chunk(dict(time=-1)).quantile(qs, dim='time', method='hazen', skipna=False)
    T_emp = T_emp.assign_coords(return_period=(np.round(1/(1-T_emp['quantile']))))
    T_emp = T_emp.swap_dims({'quantile':'return_period'})
    return T_gumb, T_gev, T_emp

def stacking_ensemble(ds_sub):
    stacked = ds_sub.stack(z=('time','realization')).reset_index('z')
    stacked = stacked.assign_coords({'z':np.arange(0,len(stacked['z']))})
    stacked = stacked.rename({'time':'old_time'})
    stacked = stacked.rename_dims({'z':'time'})
    return stacked

#%% Code
if __name__ == "__main__":
    #%%
    print("in script")
    st = time.time()

    # We import the modelled data
    #Folder_start = r"p:/11208719-interreg"
    Folder_start = r"/p/11208719-interreg"
    model_wflow = "o_rwsinfo"
    Folder_p = os.path.join(Folder_start, "wflow", model_wflow)
    folder = "members_bias_corrected_revised_daily"
    fn_fig = os.path.join(Folder_start, "Figures", model_wflow, folder)
    fn_data_out = os.path.join(fn_fig, 'data')

    if not os.path.exists(fn_fig):
        os.makedirs(fn_fig)
    
    if not os.path.exists(fn_data_out):
        os.makedirs(fn_data_out)

    #%%Check how it works for multiple ensemble in xclim #########################
    fn = glob.glob(os.path.join(Folder_p, folder, '*/output.nc'))    
    Qens = create_ensemble(fn).drop_dims('layer') #For now selecting two ensemble

    #Qens = create_ensemble(fn[0:3]).drop_dims('layer') #For now selecting two ensemble
    #Qens = Qens.sel(time=slice("1950-10-01", "1955-09-30")) #Hydrological years
    #Qens = Qens.sel(lat=slice(50,50.5), lon=slice(5,5.5)) #Selecting a subsection of lat, lon for now

    #We adress the chunking issues
    Qens['Q'] = Qens['Q'].chunk(chunks={'realization':1, 'time':-1, 'lat':25, 'lon': 50}) #chunk of about 128 MiB

    Ts = [10,50,100,1000]
    qs = [1-1/10, 1-1/50, 1-1/100, 1-1/1000]

    mask_riv = Qens.isel(realization=0, time=0)['Q']>0
    mask_riv = mask_riv.drop(['time','realization'])

    mask_riv.to_netcdf(os.path.join(fn_data_out,'mask_rivers_wflow.nc'))

    print("Data loaded")

    #%% Getting the block maximas - all, summer, winter
    sub = xclim.indices.generic.select_resample_op(Qens, op='max', freq='AS-Oct')
    summer = xclim.indices.generic.select_resample_op(Qens, op='max', freq='AS-Oct', month=[4,5,6,7,8,9])
    winter = xclim.indices.generic.select_resample_op(Qens, op='max', freq='AS-Oct', month=[10,11,12,1,2,3])

    sub['Q'] = sub['Q'].chunk(chunks={'realization':1, 'time':-1, 'lat':-1, 'lon': -1}) #chunk of about 128 MiB
    summer['Q'] = summer['Q'].chunk(chunks={'realization':1, 'time':-1, 'lat':-1, 'lon': -1}) #chunk of about 128 MiB
    winter['Q'] = winter['Q'].chunk(chunks={'realization':1, 'time':-1, 'lat':-1, 'lon': -1}) #chunk of about 128 MiB


    summer.to_netcdf(os.path.join(fn_data_out,'AM_summer_4_9_Oct.nc'))
    winter.to_netcdf(os.path.join(fn_data_out,'AM_winter_10_3_Oct.nc'))

    print("Block maxima done")

    #%% Month of the extremes #https://github.com/Ouranosinc/xclim/blob/b96daa2a1d390782e71f07e930499a3b218d298b/xclim/indices/_hydrology.py#L224
    
    datasets_month=[]
    datasets_day=[]
    all_years = [i for i in sub['time'].values]
    for i in np.arange(0,len(all_years)-1,1): 
        print(all_years[i])
        print(all_years[i+1])
        da = Qens.sel(time=slice(all_years[i], all_years[i+1]))['Q'].chunk(chunks={'realization':-1, 'time':-1, 'lat':-1, 'lon': -1})  #Should be a dataArray
        max_time = da.fillna(0).argmax(dim='time').compute()
        sub_month = da.time[max_time].dt.month
        sub_month = sub_month.where(mask_riv).drop('time')
        sub_month = sub_month.assign_coords(time = i)
        sub_month = sub_month.expand_dims('time')
        
        sub_day = da.time[max_time].dt.dayofyear.where(mask_riv).drop('time')
        sub_day = sub_day.assign_coords(time = i)
        sub_day = sub_day.expand_dims('time')

        datasets_month.append(sub_month)
        datasets_day.append(sub_day)       
    sub_month = xr.concat(datasets_month, dim='time')#.to_dataset()
    sub_day = xr.concat(datasets_day, dim='time')#.to_dataset()
    sub['month'] = sub_month
    sub['dayofyear'] = sub_day

    sub.to_netcdf(os.path.join(fn_data_out,'AM_datesAM_Oct.nc'))
    print('Data saved')
    print("Block maxima dates done")

    #%%
    all_years = stacking_ensemble(sub)
    summer = stacking_ensemble(summer)
    winter = stacking_ensemble(winter)

    all_params = xclim.indices.stats.fit(all_years['Q'].chunk(chunks={'time':-1, 'lat':150, 'lon': 150}), dist="genextreme", method="ML")
    summer_params = xclim.indices.stats.fit(summer['Q'].chunk(chunks={'time':-1, 'lat':150, 'lon': 150}), dist="genextreme", method="ML")
    winter_params = xclim.indices.stats.fit(winter['Q'].chunk(chunks={'time':-1, 'lat':150, 'lon': 150}), dist="genextreme", method="ML")

    print("GEV models fitted")
    
    all_params.to_netcdf(os.path.join(fn_data_out,'GEV_year_params_1040years.nc'))
    summer_params.to_netcdf(os.path.join(fn_data_out,'GEV_summer_params_1040years.nc'))
    winter_params.to_netcdf(os.path.join(fn_data_out,'GEV_winter_params_1040years.nc'))
    print('Data saved')


    #%% calculating per ensemble

    #Return periods - Gumbel and GEV and empirical 
    T_gumb = xclim.indices.stats.fa(sub['Q'].chunk(chunks={'realization':1, 'time':-1, 'lat':-1, 'lon': -1}), t=Ts, dist="gumbel_r", mode="max")
    T_gev = xclim.indices.stats.fa(sub['Q'].chunk(chunks={'realization':1, 'time':-1, 'lat':-1, 'lon': -1}), t=Ts, dist="genextreme", mode="max")
    print('Return periods per ensemble done')

    T_gumb.to_netcdf(os.path.join(fn_data_out,'Gumbel_return_periods_per_ensemble.nc'))
    T_gev.to_netcdf(os.path.join(fn_data_out,'GEV_return_periods_per_ensemble.nc'))
    print('Data saved')

    #Parameter sensitivities across ensembles
    params = xclim.indices.stats.fit(sub['Q'].chunk(chunks={'realization':1, 'time':-1, 'lat':-1, 'lon': -1}), dist="genextreme", method="ML")
    params = params.to_dataset()
    params['median'] = params['Q'].median(dim='realization')
    params['loc_range'] = params['Q'].sel(dparams='loc').max(dim='realization') - params['Q'].sel(dparams='loc').min(dim='realization')
    params['scale_range'] = params['Q'].sel(dparams='scale').max(dim='realization') - params['Q'].sel(dparams='scale').min(dim='realization')
    params['shape_range'] = params['Q'].sel(dparams='c').max(dim='realization') - params['Q'].sel(dparams='c').min(dim='realization')
    print("Statistics per ensemble done")

    params.to_netcdf(os.path.join(fn_data_out,'GEV_year_params_per_ensemble.nc'))
    print('Data saved')

    #Looking at whether shape changes sign across ensembles
    shape_positive = xr.where(params['Q'].sel(dparams='c')>0, 1, 0)
    shape_positive_sum = shape_positive.sum(dim='realization') 
    shape_positive_sum = shape_positive_sum.where(mask_riv)
    print("Statistics per ensemble done")

    #%%Stacking for all ensembles for 1040 years
    # Return periods - all ensembles
    all_Tgumb, all_Tgev, all_Temp = spatial_T_analysis(all_years['Q'].chunk(chunks={'time':-1, 'lat':150, 'lon': 150}), Ts)
    print("Statistics 1040 years done")

    all_Tgumb.to_netcdf(os.path.join(fn_data_out,'Gumbel_return_periods_1040years.nc'))
    all_Tgev.to_netcdf(os.path.join(fn_data_out,'GEV_return_periods_1040years.nc'))
    all_Temp.to_netcdf(os.path.join(fn_data_out,'EMP_return_periods_1040years.nc'))
    print('Data saved')

    # Return periods - summer
    summer_Tgumb, summer_Tgev, summer_Temp = spatial_T_analysis(summer['Q'].chunk(chunks={'time':-1, 'lat':150, 'lon': 150}), Ts)
    print("Statistics 1040 years done")

    summer_Tgumb.to_netcdf(os.path.join(fn_data_out,'Gumbel_summer_return_periods_1040years.nc'))
    summer_Tgev.to_netcdf(os.path.join(fn_data_out,'GEV_summer_return_periods_1040years.nc'))
    summer_Temp.to_netcdf(os.path.join(fn_data_out,'EMP_summer_return_periods_1040years.nc'))
    print('Data saved')

    # Return periods - winter
    winter_Tgumb, winter_Tgev, winter_Temp = spatial_T_analysis(winter['Q'].chunk(chunks={'time':-1, 'lat':150, 'lon': 150}), Ts)
    print("Statistics 1040 years done")

    winter_Tgumb.to_netcdf(os.path.join(fn_data_out,'Gumbel_winter_return_periods_1040years.nc'))
    winter_Tgev.to_netcdf(os.path.join(fn_data_out,'GEV_winter_return_periods_1040years.nc'))
    winter_Temp.to_netcdf(os.path.join(fn_data_out,'EMP_winter_return_periods_1040years.nc'))
    print('Data saved')


    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

#%% OLD SCRIPT
# sub = select_resample_op(pr, op="max", freq="YS", month=[5, 6, 7, 8, 9, 10])

#%%
# extremes.get_peaks(da1.fillna(0), ev_type="BM", period = "365.25D")
# # Using hydromt
# da = da1.isel(time=slice(0,365*5))
# # 
# da = da.load()
# # - very slow! Will be problematic!
# bm_peaks = extremes.get_peaks(da, ev_type="BM", period = "365.25D")
# da_params = extremes.fit_extremes(bm_peaks, ev_type="BM", distribution="gev")



# #Check this
# datasets = []
# for i in np.arange(0,len(Qens['realization'])):
#     print(i)
#     da = Qens.sel(realization=i)['Q'] #Should be a dataArray
#     da = da.where(da>0,0)
#     sub_day = xclim.indices.generic.select_resample_op(da.load(), op=xclim.indices.generic.doymax, freq='AS-OCT')
#     sub_day = sub_day.where(mask_riv)
#     datasets.append(sub_day)
#     print('done')
# sub_month = xr.concat(datasets, dim='realization')#.to_dataset()
# sub['dayofyear'] = sub_month

    #%% Using xclim on one member ######################################### 
    # fn = os.path.join(Folder_p, folder, 'r10i1p5f1', 'output.nc')
    # ds = xr.open_dataset(fn, chunks='auto') #, chunks={'time':2000})
    # da1 = ds['Q'].sel(time=slice("1950-10-01", "1955-09-30")) #We select hydrological years
    # da1 = da1.sel(lat=slice(50,50.5), lon=slice(5,5.5)) # select a subset of lat, lon

    # #1-We extract the maximum
    # sub = xclim.indices.generic.select_resample_op(da1, op='max', freq='AS-Oct')
    # #2-We fit the parameters
    # params = xclim.indices.stats.fit(sub.chunk(dict(time=-1)), dist="gumbel_r", method="ML")
    # #3-We extract the quantile
    # T_100_gumbel = xclim.indices.stats.parametric_quantile(params, q=1 - 1.0 / 100)
    # #The last two functions 2&3 can be combined in:
    # T100_gumb = xclim.indices.stats.fa(sub.chunk(dict(time=-1)), t=[20,50,100], dist="gumbel_r", mode="max")
