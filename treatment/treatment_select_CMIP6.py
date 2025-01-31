import time as ti

import numpy as np
import xarray as xr
from fcts_support_select import *

from fcts_support_io import *
from fcts_support_taylorDiagram import *

#==================================================================================================
# 0. OPTIONS
#==================================================================================================
# general info
period_seasons = [1950, 2020] # period for comparison of seasonalities

# criteria for selection - HERE, ONLY FOR PLOTS!
threshold_std = 0.05 # std-dev within this fraction of obs
threshold_coerr = 0.95 # correlation coefficient with obs higher than this value

# info on CMIP6 files
var_input_CMIP6 = 'tas'
list_experiments_CMIP6 = ['historical','ssp245'] # 1 historical, 1 ssp
list_members_CMIP6 = ['r1i1p1f1']

# preparing paths
paths_in = {'select_CMIP6':'/landclim/yquilcaille/contributions_FF/select_CMIP6',\
            'CMIP6ng':'/net/ch4/data/cmip6-Next_Generation'}
#==================================================================================================
#==================================================================================================





#==================================================================================================
# 1. SEASONALITY ERA5
#==================================================================================================
# file to use for seasonality
path_era5_seasonality = os.path.join(paths_in['select_CMIP6'], 'ERA5_seasonality-'+str(period_seasons[0])+'-'+str(period_seasons[1])+'_g025.nc')

# checking its existence
if os.path.isfile( path_era5_seasonality ):
    print('loading seasonality of ERA5')
    # loading seasonality of observations
    obs_season = xr.open_dataset( path_era5_seasonality )['t2m']
    
else:
    # loading observations
    print('loading ERA5')
    era5_g025 = xr.open_dataset( os.path.join( paths_in['select_CMIP6'], 'ERA5_g025.nc' ) )

    # evaluate seasonality of obs if not calculated yet
    print('calculating seasonality of ERA5')
    obs_season = eval_seasonality( ds=era5_g025['t2m'], period=period_seasons )

    # save that
    tmp = xr.Dataset()
    tmp['t2m'] = obs_season
    tmp.to_netcdf( path_era5_seasonality, encoding={var: {"zlib": True} for var in tmp.variables} )
#==================================================================================================
#==================================================================================================




#==================================================================================================
# 2. SEASONALITY CMIP6-ng
#==================================================================================================
# Loading CMIP6-ng
cmip6ng = files_cmip6ng(var_input=var_input_CMIP6, list_experiments=list_experiments_CMIP6, list_members=list_members_CMIP6, path_cmip6ng=paths_in['CMIP6ng'])
cmip6ng.load_all()
list_esms = list(cmip6ng.data_esm.keys())

# looping on ESMs
esm_season = {}
for esm in list_esms:
    # file to use for seasonality
    path_esm_seasonality = os.path.join(paths_in['select_CMIP6'], esm+'_seasonality-'+str(period_seasons[0])+'-'+str(period_seasons[1])+'_g025.nc')

    # checking its existence
    if os.path.isfile( path_esm_seasonality ):
        # loading seasonality of ESM
        print('loading seasonality of '+esm)
        esm_season[esm] = xr.open_dataset( path_esm_seasonality )['tas']

    else:
        # evaluate seasonality of ESMs if not calculated yet
        print('calculating seasonality of '+esm)
        esm_season[esm] = eval_seasonality( ds=cmip6ng.data_esm[esm], period=period_seasons )

        # save that
        tmp = xr.Dataset()
        tmp['tas'] = esm_season[esm]
        tmp.to_netcdf( path_esm_seasonality, encoding={var: {"zlib": True} for var in tmp.variables} )
#==================================================================================================
#==================================================================================================






#==================================================================================================
# 3. CALCULATING DISCREPANCIES BETWEEN SEASONALITIES
#==================================================================================================
# preparation
data_obs = obs_season - obs_season.mean('dayofyear')

# preparing dataset that will be saved
OUT = xr.Dataset()
OUT['stddev_ref'] = data_obs.std( dim='dayofyear', ddof=1 )
OUT['stddev'] = xr.DataArray( np.nan, coords={'esm':list_esms, 'lat':obs_season.lat, 'lon':obs_season.lon}, dims=('esm', 'lat','lon',) )
OUT['corrcoef'] = xr.DataArray( np.nan, coords={'esm':list_esms, 'lat':obs_season.lat, 'lon':obs_season.lon}, dims=('esm', 'lat','lon',) )
    
# comparing all of these seasonalities
for esm in list_esms:
    print('Comparing seasonalities of '+esm)
    _, OUT['stddev'].loc[{'esm':esm}], OUT['corrcoef'].loc[{'esm':esm}] = compare_seasonalities( data_obs=data_obs, data_esm=esm_season[esm] - esm_season[esm].mean('dayofyear') )

# quick check
if np.any(np.isnan(OUT['stddev'].values)) or np.any(np.isnan(OUT['corrcoef'].values)):
    raise Exception("NaN detected!")
    
# saving these results
OUT.to_netcdf( os.path.join(paths_in['select_CMIP6'], 'comparison_seasonalities-'+str(period_seasons[0])+'-'+str(period_seasons[1])+'_g025.nc'), encoding={var: {"zlib": True} for var in ['stddev_ref', 'stddev','corrcoef']} )
#==================================================================================================
#==================================================================================================





#==================================================================================================
# 4. PLOTS
#==================================================================================================
#-------------------------------------------------------------
# 4.1. Taylor plot
#-------------------------------------------------------------
if False:
    # NB: using this part on a loop latitude/longitude takes way too long to produce seasonalities / select ESMs.
    # solutions tried to reduce that: 'do_taylor' with only values (ie 'compare_seasonalities' only), no plots; parallel computing.
    # yet, still too long because of number of grid cells.
    # keep this part ONLY for plots.
    i_lat, i_lon = 50, 100
    data_obs = obs_season - obs_season.mean('time')
    data_esm = {esm: esm_season[esm] - esm_season[esm].mean('time') for esm in list_esms}
    dia, fig, refstd, dico_stddev, dico_corrcoef, selected = do_taylor(data_obs=data_obs, data_esm=data_esm, threshold_std=threshold_std, threshold_coerr=threshold_coerr,\
                                                                       lat_gp=era5_g025.lat.values[i_lat], lon_gp=era5_g025.lon.values[i_lon],\
                                                                       labels={'X':'Day of year', 'Y':'Anomaly in daily temperature ('+ u'\u00B0C'+')'}, ncols_lgd=3, fact_sizes=2, sorted_colors=False )
#-------------------------------------------------------------
#-------------------------------------------------------------




#-------------------------------------------------------------
# 4.2. Map for an ESM of where kept
#-------------------------------------------------------------
if False:
    # which one to plot?
    esm = list_esms[0]

    # initialize plot
    fig = plt.figure( figsize=(20,20) )
    
    # Map of standard deviation
    ax = plt.subplot( 311, projection=ccrs.Robinson() )
    plt.title( 'Standard deviation of '+esm, size=15 )
    ax, pmesh = func_map( data_plot=OUT['stddev'].sel(esm=esm).values, ax=ax, lat=OUT.lat.values, lon=OUT.lon.values, fontsize_colorbar=12, vmin=None, vmax=None, n_levels=100, symetric_colorbar=False )
    
    # Map of correlation coefficient
    ax = plt.subplot( 312, projection=ccrs.Robinson() )
    plt.title( 'Correlation coefficient of '+esm, size=15 )
    ax, pmesh = func_map( data_plot=OUT['corrcoef'].sel(esm=esm).values, ax=ax, lat=OUT.lat.values, lon=OUT.lon.values, fontsize_colorbar=12, vmin=None, vmax=None, n_levels=100, symetric_colorbar=True )
    
    # Map of selection
    ax = plt.subplot( 313, projection=ccrs.Robinson() )
    kept = xr.where( (OUT['stddev'].sel(esm=esm) - OUT['stddev_ref'] <= threshold_std * OUT['stddev_ref']) & (OUT['corrcoef'].sel(esm=esm) >= threshold_coerr), 1, 0 )
    plt.title( 'Selection of '+esm, size=15 )
    ax, pmesh = func_map( data_plot=kept.values, ax=ax, lat=OUT.lat.values, lon=OUT.lon.values, fontsize_colorbar=12, vmin=None, vmax=None, n_levels=2, symetric_colorbar=False )
#-------------------------------------------------------------
#-------------------------------------------------------------




#-------------------------------------------------------------
# 4.3. Map of number of ESMs by grid point
#-------------------------------------------------------------
if False:
    threshold_std = 0.05 # std-dev within this fraction of obs
    threshold_coerr = 0.95 # correlation coefficient with obs higher than this value
    
    # checking
    kept_stddev = xr.where( (OUT['stddev'] - OUT['stddev_ref'] <= threshold_std * OUT['stddev_ref']), 1, 0 ).sum('esm')
    kept_corr = xr.where(  (OUT['corrcoef'] >= threshold_coerr), 1, 0 ).sum('esm')
    kept = xr.where( (np.abs( OUT['stddev'] - OUT['stddev_ref'] ) <= threshold_std * OUT['stddev_ref']) & (OUT['corrcoef'] >= threshold_coerr), 1, 0 ).sum('esm')
    
    # initialize plot
    fig = plt.figure( figsize=(20,30) )
    
    # standard deviation
    ax = plt.subplot( 311, projection=ccrs.Robinson() )
    plt.title( 'Number of ESMs kept due to std dev (range: '+str(kept_stddev.min().values)+' - '+ str(kept_stddev.max().values) +')', size=15 )
    ax, pmesh = func_map( data_plot=kept_stddev.values, ax=ax, lat=OUT.lat.values, lon=OUT.lon.values, fontsize_colorbar=12, vmin=0, vmax=OUT.esm.size, n_levels=OUT.esm.size+1, symetric_colorbar=False )
    
    # correlation coefficient
    ax = plt.subplot( 312, projection=ccrs.Robinson() )
    plt.title( 'Number of ESMs kept due to corr (range: '+str(kept_corr.min().values)+' - '+ str(kept_corr.max().values) +')', size=15 )
    ax, pmesh = func_map( data_plot=kept_corr.values, ax=ax, lat=OUT.lat.values, lon=OUT.lon.values, fontsize_colorbar=12, vmin=0, vmax=OUT.esm.size, n_levels=OUT.esm.size+1, symetric_colorbar=False )
    # OUT['corrcoef'].max('esm').min() ~ 0.34...... => cannot lower that much

    # both
    ax = plt.subplot( 313, projection=ccrs.Robinson() )
    plt.title( 'Number of ESMs kept (range: '+str(kept.min().values)+' - '+ str(kept.max().values) +')', size=15 )
    ax, pmesh = func_map( data_plot=kept.values, ax=ax, lat=OUT.lat.values, lon=OUT.lon.values, fontsize_colorbar=12, vmin=0, vmax=OUT.esm.size, n_levels=OUT.esm.size+1, symetric_colorbar=False )
#-------------------------------------------------------------
#-------------------------------------------------------------
#==================================================================================================
#==================================================================================================




