import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from fcts_support_io import *

path_GMT = '/net/exo/landclim/yquilcaille/contributions_FF/GMT'


#--------------------------------------------------------------------------------------------------------------------------------------------
# 1. BEST
#--------------------------------------------------------------------------------------------------------------------------------------------
# both options possible
options_sea_ice_temp_from = ['air_temperatures', 'water_temperatures']

# preparing output
OUT_BEST = xr.Dataset()
OUT_BEST.coords['sea_ice_temp_from'] = ['air_temperatures', 'water_temperatures']
OUT_BEST.coords['time'] = np.arange( 1850, 2022+1 )
OUT_BEST['temperature'] = xr.DataArray( np.nan, coords={'time':yrs, 'sea_ice_temp_from':options_sea_ice_temp_from}, dims=('time','sea_ice_temp_from', ) )
OUT_BEST['temperature_uncertainty'] = xr.DataArray( np.nan, coords={'time':yrs, 'sea_ice_temp_from':options_sea_ice_temp_from}, dims=('time','sea_ice_temp_from', ) )


# loading monthly BEST -- daily BEST goes up to 31.07.2022, while they succeeded in evaluating the monthly up to end of 
with open( os.path.join(path_GMT,'Global_BEST_land-ocean_1850-2023.txt'), newline='') as csvfile:
    read = csv.reader(csvfile, delimiter=' ')
    best = [row for row in read]

for sea_ice_temp_from in options_sea_ice_temp_from:
    # reading lines
    indexes = {'air_temperatures':[86,2164], 'water_temperatures':[2172, 4250] }[sea_ice_temp_from]
    yrs, temperature_monthly, temperature_monthly_unc = [], [], []
    for row in best[indexes[0]:indexes[1]+1]:
        # removing empty spaces
        while '' in row:
            row.remove( '' )
        # time of year
        yr, m = int(row[0]), int(row[1])

        # number of days for weights
        d = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31 }[m]
        if (yr % 4 == 0):
            tot_d = 366
            if (m == 2):
                d += 1
        else:
            tot_d = 365

        # preparing annual average
        if m == 1:
            tmp_monthly = float(row[2]) * d / tot_d
            tmp_monthly_unc = ( float(row[4]) * d / tot_d ) ** 2
        else:
            tmp_monthly += float(row[2]) * d / tot_d
            tmp_monthly_unc += ( float(row[4]) * d / tot_d ) ** 2
        # saving average
        if m == 12:
            yrs.append( yr )
            temperature_monthly.append( tmp_monthly )
            temperature_monthly_unc.append( np.sqrt( tmp_monthly_unc ) )
        
    # allocating that
    OUT_BEST['temperature'].loc[{'sea_ice_temp_from':sea_ice_temp_from}] = temperature_monthly
    OUT_BEST['temperature_uncertainty'].loc[{'sea_ice_temp_from':sea_ice_temp_from}] = temperature_monthly_unc
    
# saving data
OUT_BEST.to_netcdf( os.path.join(path_GMT,'BEST_GMT.nc'), encoding={var: {"zlib": True} for var in OUT_BEST.variables} )
#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------------------------------------------------
# 2. ERA5
#--------------------------------------------------------------------------------------------------------------------------------------------
path_era5 = '/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_1h/processed/regrid_tmean1d'

# finding files
files = [file for file in os.listdir( path_era5 ) if (file[-3:]=='.nc') and (file.split('.')[1]=='t2m')]

# loading files
era5 = xr.open_mfdataset( [os.path.join(path_era5, file) for file in files], preprocess=preprocess_ERA5, combine='by_coords'  )

# spatial average
weights = np.cos(np.deg2rad(era5.lat))
tmp = era5['t2m'].weighted(weights).mean(dim=("lat", "lon"))

# annual average --> works for dataset ERA5
rsmpl = tmp.resample(time="1Y")
i_yrs_kept = []
for i, gp in enumerate(rsmpl.groups):
    stop, start = rsmpl.groups[gp].stop, rsmpl.groups[gp].start
    if (stop is not None):
        if (start is not None) and (stop - start) >= 365:
            i_yrs_kept.append( i )
    elif (start is not None) and (tmp.time.size - start) >= 365:
        i_yrs_kept.append( i )
values = rsmpl.mean()
values.coords['time'] = np.array( pd.DatetimeIndex(values.time).year )

# compute everything
gmt_era5 = values.isel( time=i_yrs_kept ).compute()

# preparation before save
OUT_ERA5 = xr.Dataset()
OUT_ERA5['t2m'] = gmt_era5# - 273.15 # no need for that anymore
OUT_ERA5['t2m'].attrs['unit'] = 'degC'

# saving data
OUT_ERA5.to_netcdf( os.path.join(path_GMT,'ERA5_GMT.nc'), encoding={var: {"zlib": True} for var in OUT_ERA5.variables} )

# quick comparison
if False:
    import matplotlib.pytplot as plt
    
    plt.figure( figsize=(15,10) )
    ( OUT_ERA5['t2m'] - OUT_ERA5['t2m'].sel(time=slice(1950,1970)).mean('time') ).plot( label='ERA5' )
    for sea_ice_temp_from in OUT_BEST.sea_ice_temp_from.values:
        ( OUT_BEST['temperature'] - OUT_BEST['temperature'].sel(time=slice(1950,1970)).mean('time') ).sel(sea_ice_temp_from=sea_ice_temp_from).plot( label='BEST-'+sea_ice_temp_from )
    plt.grid()
    plt.legend(loc=0)
#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------




#--------------------------------------------------------------------------------------------------------------------------------------------
# 3. CMIP6
#--------------------------------------------------------------------------------------------------------------------------------------------
var_input_CMIP6 = 'tas'
list_experiments_CMIP6 = ['historical','ssp245'] # 1 historical, 1 ssp
list_members_CMIP6 = ['r1i1p1f1']
path_cmip6ng = '/net/ch4/data/cmip6-Next_Generation'

# Loading CMIP6-ng
cmip6ng = files_cmip6ng(var_input=var_input_CMIP6, list_experiments=list_experiments_CMIP6, list_members=list_members_CMIP6, path_cmip6ng=path_cmip6ng)
cmip6ng.load_all()
list_esms = list(cmip6ng.data.keys())

# calculating GMT of the ESM: spatial average
for esm in list_esms:
    print(esm)
    weights = np.cos(np.deg2rad(cmip6ng.data[esm].lat))
    tmp = cmip6ng.data[esm].weighted(weights).mean(dim=('lat', 'lon'))
    
    # annual average
    gmt_esm = tmp.resample(time="1Y").mean().compute()
    
    # save that
    GMT = xr.Dataset()
    GMT['tas'] = gmt_esm
    GMT['tas'].attrs['experiments'] = str(list_experiments_CMIP6)
    GMT.to_netcdf( os.path.join(path_GMT,esm+'_GMT.nc'), encoding={var: {"zlib": True} for var in GMT.variables} )
#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------


