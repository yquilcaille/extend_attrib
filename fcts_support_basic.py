import os

import cftime as cft
import numpy as np
import xarray as xr


def eval_seasonality( ds, period ):
    # remove useless dimensions for this precise case
    for crd in ['esm', 'member']:
        if crd in ds.coords:
            ds = ds.loc[{crd:ds[crd].values[0]}].drop(crd)
    
    # selection in time
    val0 = ds.time.values[0]
    t0 = format_date( dict_date={'Year':period[0],'Month':1,'Day':1}, val0=val0 )
    t1 = format_date( dict_date={'Year':period[1],'Month':12,'Day':31}, val0=val0 )
    ds = ds.sel( time=slice(t0,t1) )
        
    # seasonality
    vals = ds.groupby('time.dayofyear').mean().compute()
    # drop 29th february if there (depends formats). useful to compare to climatology of ERA5 which is over 365 days.
    if vals.dayofyear.size == 366:
        tt = list( np.arange(366) )
        tt.remove( 31+29 )
        vals = vals.isel( dayofyear=tt )
    return vals # xr.DataArray( vals.values, coords={'dayofyear':np.arange(vals.dayofyear.size)}, dims=('dayofyear',) )

    

def format_date( dict_date, val0=None ):

    # creating date
    if (type(val0) == np.datetime64) or (val0 is None):
        # preparing date
        dic_dt = {}
        for dt in ['Year', 'Month', 'Day']:
            dic_dt[dt] = str(int(dict_date[dt]))
            if len(dic_dt[dt]) == 1:
                dic_dt[dt] = '0'+dic_dt[dt]
        return np.datetime64( '-'.join( [dic_dt['Year'], dic_dt['Month'], dic_dt['Day']] ) + 'T' + str(val0).split('T')[1], 'D')
    else:
        format_t = type(val0)
        if format_t in [cft._cftime.Datetime360Day] and dict_date['Day'] == 31:
            return format_t( year=dict_date['Year'], month=dict_date['Month'], day=30, hour=val0.hour, minute=val0.minute, second=val0.second, microsecond=val0.microsecond)
        else:
            return format_t( year=dict_date['Year'], month=dict_date['Month'], day=dict_date['Day'], hour=val0.hour, minute=val0.minute, second=val0.second, microsecond=val0.microsecond)


        

def compare_seasonalities( data_obs, data_esm ):
    if data_esm.ndim == 1:
        # case used for do_Taylor: for 1 grid point
        # check related to time formats
        if data_esm.dayofyear.size != data_obs.dayofyear.size:
            data_esm2 = np.interp( x=data_obs.dayofyear.values, xp=data_esm.dayofyear.values * data_obs.dayofyear.size/data_esm.dayofyear.size, fp=data_esm.values )
        else:
            data_esm2 = data_esm
        stddev = data_esm2.std( dim='dayofyear', ddof=1 )
        corrcoef = xr.corr( da_a=data_esm2, da_b=data_obs, dim='dayofyear' )
        #stddev = np.std( data_esm2, ddof=1 )
        #corrcoef = np.corrcoef( data_obs.values, data_esm2 )[0,1]
        return data_esm2, stddev, corrcoef

    else:
        # case used for production of results over whole map
        
        # check related to time formats
        if data_esm.dayofyear.size != data_obs.dayofyear.size:
            data_esm2 = xr.DataArray( np.nan, coords={'dayofyear':data_obs.dayofyear , 'lat':data_esm.lat, 'lon':data_esm.lon}, dims=('dayofyear', 'lat', 'lon',) )
            x = data_obs.dayofyear.values
            xp = data_esm.dayofyear.values * data_obs.dayofyear.size/data_esm.dayofyear.size
            for lat in data_esm.lat.values:
                for lon in data_esm.lon.values:
                    data_esm2.loc[{'lat':lat,'lon':lon}] = np.interp( x=x, xp=xp, fp=data_esm.sel(lat=lat,lon=lon).values )
        else:
            data_esm2 = data_esm
        
        # looping on latitudes & longitudes
        stddev = data_esm2.std( dim='dayofyear', ddof=1 )
        corrcoef = xr.corr( da_a=data_esm2, da_b=data_obs, dim='dayofyear' )
        return data_esm2, stddev, corrcoef
        
def select_cmip6(path, period_seasons ):
    # open file for comparison
    comp_cmip6ng = xr.open_dataset( os.path.join(path, 'comparison_seasonalities-'+str(period_seasons[0])+'-'+str(period_seasons[1])+'_g025.nc') )
    
    # evaluate the mask
    #threshold_std, threshold_coerr
    #kept = xr.where( (np.abs( comp_cmip6ng['stddev'] - comp_cmip6ng['stddev_ref'] ) <= threshold_std * comp_cmip6ng['stddev_ref']) & (comp_cmip6ng['corrcoef'] >= threshold_coerr), 1, 0 )
    return comp_cmip6ng


def weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    
    # Souce: https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    if np.any( quantiles < 0 ) or np.any(quantiles > 1):
        raise Exception('quantiles should be in [0, 1]')

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)