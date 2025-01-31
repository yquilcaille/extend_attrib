import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

#----------------------------------------------------------------------------------------------------------
# OPTIONS TREATMENT
#----------------------------------------------------------------------------------------------------------
path_CEDS = '/net/exo/landclim/yquilcaille/contributions_FF/CEDS'
name_files = {'CH4-air-1970-2014': 'CH4-em-AIR-anthro_input4MIPs_emissions_CMIP_CEDS-2017-08-30_gn_197001-201412.nc',\
              'CH4-air-1850-1960': 'CH4-em-AIR-anthro_input4MIPs_emissions_CMIP_CEDS-2017-08-30-supplemental-data_gn_185001-196012.nc',\
              'CH4-all-1970-2014': 'CH4-em-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18_gn_197001-201412.nc',\
              'CH4-all-1850-1960': 'CH4-em-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18-supplemental-data_gn_185001-196012.nc',\
              'CH4-bio-1970-2014': 'CH4-em-SOLID-BIOFUEL-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18-supplemental-data_gn_197001-201412.nc'}

path_Jones023 = '/net/exo/landclim/yquilcaille/contributions_FF/Jones023'
#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------
# CEDS
#----------------------------------------------------------------------------------------------------------
# load
data = {}
for file in name_files:
    data[file] = xr.open_dataset( os.path.join( path_CEDS, name_files[file] ) )
# cf comments in files, stick only to 'all':

# treat
data_treated = {}
for file in ['CH4-all-1850-1960', 'CH4-all-1970-2014']:
    # get variable: kg m-2 s-1
    tmp = data[file]['CH4_em_anthro']
    
    # sum over sectors
    tmp = tmp.sum('sector')
    
    # sum over time (Calendar: DateTimeNoLeap)
    tmp = tmp.resample(time="Y").sum() * 365/12 * 24 * 3600
    
    # sum over longitude latitude
    Surf_Earth = 509600000 * 1.e6
    weights = np.cos(np.deg2rad(tmp.lat))
    data_treated[file] = tmp.weighted(weights).mean(dim=("lat", "lon")) * Surf_Earth * 1.e-9 # Mt CH4 yr-1
    
    # adapting coord time
    data_treated[file].coords['time'] = data_treated[file].time.dt.year

# creating final data
OUT_CEDS = xr.Dataset()
OUT_CEDS['emissions_CH4'] = xr.concat( [data_treated['CH4-all-1850-1960'], data_treated['CH4-all-1970-2014']], dim='time' )
OUT_CEDS['emissions_CH4'] = OUT_CEDS['emissions_CH4'].interpolate_na(dim='time', method='linear')

# check
if False:
    plt.scatter( data_treated['CH4-all-1850-1960'].time, data_treated['CH4-all-1850-1960'] )
    plt.plot( data_treated['CH4-all-1970-2014'].time, data_treated['CH4-all-1970-2014'] )
    plt.plot( OUT_CEDS.time, OUT_CEDS['emissions_CH4'], color='k', ls='--' )
    
# saving
OUT_CEDS = OUT_CEDS.rename({'time':'year'})
OUT_CEDS.to_netcdf( os.path.join(path_CEDS, 'emissions_CEDS-CH4_1850-2014.nc'), encoding={var: {"zlib": True} for var in OUT_CEDS.variables} )
#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------------------
# Jones et al, 2023 (https://doi.org/10.1038/s41597-023-02041-1)
#----------------------------------------------------------------------------------------------------------
with open(os.path.join(path_Jones023, 'EMISSIONS_ANNUAL_1830-2021.csv'), newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    dico_vals = {}
    tmp_gas, tmp_comp, tmp_unit, tmp_country = [], [], [], [] # check
    for row in reader:
        if row[0] not in ['"CNTR_NAME"']:
            # separation of rows in csv based on ',', but commas are also present in names of countries, implying changing index of rows...

            # prepare data
            country, iso, gas, comp, year, val, unit = row[:-7+1], row[-6], row[-5], row[-4], row[-3], row[-2], row[-1]
            
            # checking gas and component
            if gas not in tmp_gas:
                tmp_gas.append(gas)
            if comp not in tmp_comp:
                tmp_comp.append(comp)
            if unit not in tmp_unit:
                tmp_unit.append(unit)
            if country not in tmp_country:
                tmp_country.append(country)
            # ok, no issues with multiple names
            # different categories (country, region, groups of countries), taking only '"GLOBAL"'

            # check whether keeps that one:
            if (gas=='"CH[4]"')  and  (comp in ['"Fossil"', '"LULUCF"'])  and  (country==['"GLOBAL"']):
                # treat iso, comp, year
                iso = eval(iso)
                year = eval(year)
                comp = eval(comp)
                
                # treat val
                if val in ['NA']:
                    val = np.nan
                else:
                    val = eval(val)
        
                # sorting data
                if iso not in dico_vals:
                    dico_vals[iso] = {}
                if comp not in dico_vals[iso]:
                    dico_vals[iso][comp] = {}
                dico_vals[iso][comp][year] = val

# preparing better format
isos = list(dico_vals.keys())
comps = ['Fossil', 'LULUCF']
years = np.arange(1830, 2021+1)
tmp = np.nan * np.ones( (len(isos), len(comps), len(years)) )
for i, iso in enumerate(isos):
    for c, comp in enumerate(comps):
        if comp in dico_vals[iso]:
            tmp[i,c,:] = np.array([dico_vals[iso][comp][year] for year in years])
        else:
            tmp[i,c,:] = np.nan * np.ones( len(years) )

# putting in better format
OUT_Jones2023 = xr.Dataset()
OUT_Jones2023.coords['time'] = years
for c, comp in enumerate(comps):
    OUT_Jones2023['emissions_CH4_'+comp] = xr.DataArray( np.nansum(tmp[:,c,:], axis=0), dims=('time',) )

# saving
OUT_Jones2023 = OUT_Jones2023.rename({'time':'year'})
OUT_Jones2023.to_netcdf( os.path.join(path_Jones023, 'emissions_Jones2023-CH4_1830-2021.nc'), encoding={var: {"zlib": True} for var in OUT_Jones2023.variables} )
#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------

# check
if False:
    plt.plot( OUT_Jones2023.year, OUT_Jones2023['emissions_CH4'], color='k', ls='-', label='CEDS' )
    plt.plot( OUT_Jones2023.year, OUT_Jones2023['emissions_CH4_Fossil']+OUT_Jones2023['emissions_CH4_LULUCF'], color='r', ls='--', label='Jones2023' )
    plt.grid()
    plt.legend(loc=0)






