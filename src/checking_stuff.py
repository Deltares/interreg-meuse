import xarray as xr

fn = r'/u/couasnon/pet.KNMI-2014.KEXT12.kR2v3-v578-fECEARTH3-ds23-r1i1p5f1+hist.h.nc'
ds = xr.open_dataset(fn)
ds = ds.rename_vars({'__xarray_dataarray_variable__':'pet'})
ds.attrs['units'] = 'kg/m2/s'
ds.attrs['version'] = 'Faulty year - manual addition from Leon in May 2022'
#f"full_ds/{member_number}/{var_name}/{var_name}.KNMI-{year_name}.{member_number}.nc")
ds.to_netcdf('/p/11208719-interreg/data/racmo/members_bias_corrected/a_raw/hourly/full_ds/r1i1p5f1/pet/pet.KNMI-2014.r1i1p5f1.nc')

fn2 = r'/p/11208719-interreg/data/racmo/members_bias_corrected/a_raw/hourly/full_ds/r2i1p5f1/pet/pet.KNMI-2014.r2i1p5f1.nc'
ds2=xr.open_dataset(fn2)


ds3 = xr.open_dataset(r'/p/11208719-interreg/data/racmo/members_bias_corrected/a_raw/hourly/full_ds/r1i1p5f1/pet/pet.KNMI-2014.r1i1p5f1_old.nc')
