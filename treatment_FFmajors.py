'''
WARNING: THE DATA TREATED IN THIS FILE IS COVERED BY A COPYRIGHT (Climate Accountability Institute).
THE DATA WAS PROVIDED BY RICHARD HEEDE TO YANN QUILCAILLE, AND IS NOT MEANT FOR DISTRIBUTION.
TO AVOID COPYRIGHT INFRINGEMENT, PLEASE DO NOT USE IN ANY WAY THE INFORMATION TREATED IN THIS FILE OR THE ASSOCIATED DATA.
INSTEAD, PLEASE CONTACT YANN QUILCAILLE (yann.quilcaille@env.ethz.ch).
'''

import csv
import os

import numpy as np
import xarray as xr

# TREATMENT OF THE FOSSIL-FUEL DATA PROVIDED BY RICK HEEDE TO YANN QUILCAILLE ON 14.07.2023
# THIS DATA HAS BEEN REPLACED WITH AN UPDATE BY EMMET CONNAIRE ON 27.03.2024


# preparing variables
variables_coords = ['year', 'parent_entity', 'parent_type', 'reporting_entity', 'commodity']
variables_output = ['product_emissions_MtCO2', 'flaring_emissions_MtCO2', 'venting_emissions_MtCO2', 'own_fuel_use_emissions_MtCO2',\
                    'fugitive_methane_emissions_MtCO2e', 'fugitive_methane_emissions_MtCH4', 'total_operational_emissions_MtCO2e', 'total_emissions_MtCO2e']
variables_leftout = ['production_value', 'production_unit', 'source']

# reading dataset
with open(os.path.join('/landclim/yquilcaille/contributions_FF/FF', 'emissions_high_granularity.csv'), newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    tmp_data = {var:[] for var in variables_coords + variables_output}
    for ir, row in enumerate(reader):
        if ir == 0:
            header = row
            for i, elt in enumerate(row):
                tmp_data[elt] = []
        else:
            # issue: ',' is used BOTH to separate elements and inside elements.
            # solution: not a perfect one: year ok, then given found reasons for ','
            # reason: Inc in from parent_type to commodity
            if row[header.index('commodity')] == ' Inc."':
                row = row[:header.index('reporting_entity')] + [row[header.index('reporting_entity')]+', '+row[header.index('commodity')]] + row[header.index('commodity')+1:]
            # reason: source
            # here, it is a mess, several variants. at the end, so ignoring.

            for var in variables_coords+variables_output:
                tmp_data[var].append(row[header.index(var)])

# preparing coordinates
coords = {}
for var in variables_coords:
    if var == 'year':
        coords['year'] = np.arange(1850, 2022+1)
    else:
        coords[var] = list(set(tmp_data[var]))

# preparing dataset
data_FF = xr.Dataset(coords=coords)
for var in variables_output:
    data_FF[var] = xr.DataArray(0., coords=coords, dims=tuple(coords.keys()) )

# filling in dataset
for i in np.arange( len(tmp_data['year']) ):
    # identifying point
    tmp_coords = {c: tmp_data[c][i] for c in coords.keys()}
    tmp_coords['year'] = int(tmp_coords['year'])
    # fill in variables
    for var in variables_output:
        data_FF[var].loc[tmp_coords] = data_FF[var].loc[tmp_coords] + float(tmp_data[var][i])

# summarizing dataset
data_FF_v2 = xr.Dataset()
data_FF_v2['emissions_CO2'] = (data_FF['product_emissions_MtCO2'] + data_FF['flaring_emissions_MtCO2'] + data_FF['venting_emissions_MtCO2'] + data_FF['own_fuel_use_emissions_MtCO2']).sum( ('commodity', 'parent_type', 'reporting_entity') )
data_FF_v2['emissions_CH4'] = (data_FF['fugitive_methane_emissions_MtCH4']).sum( ('commodity', 'parent_type', 'reporting_entity') )

# renaming axis entity
data_FF_v2 = data_FF_v2.rename({'parent_entity': 'entity'})

# adding information on status entity
dico_status_entity = {}
for i in np.arange( len(tmp_data['year']) ):
    ent, sta = tmp_data['parent_entity'][i], tmp_data['parent_type'][i]
    if ent not in dico_status_entity:
        dico_status_entity[ent] = []
    if sta not in dico_status_entity[ent]:
        dico_status_entity[ent].append(sta)
if np.any( [len(dico_status_entity[ent]) > 1 for ent in dico_status_entity] ):
    raise Exception('error on status entities')
else:
    data_FF_v2['status_entity'] = xr.DataArray( [dico_status_entity[ent][0] for ent in data_FF_v2.entity.values], dims=('entity',) )

# adding information on country entity
with open(os.path.join('/landclim/yquilcaille/contributions_FF/FF', 'matching_carbon-majors_countries.csv'), newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';', quotechar='|')
    dict_matching = {eval(line[0]):eval(line[1]) for line in reader}
data_FF_v2['country_entity'] = xr.DataArray( [dict_matching[ent] for ent in data_FF_v2.entity.values], dims=('entity',) )

# preparing data
data_FF_v2['emissions_CO2'].attrs['unit'] = 'MtCO2 / yr'
data_FF_v2['emissions_CO2'].attrs['copyright'] = 'Copyright Climate Accountability Institute / Influence Map'
data_FF_v2['emissions_CH4'].attrs['unit'] = 'MtCH4 / yr'
data_FF_v2['emissions_CH4'].attrs['copyright'] = 'Copyright Climate Accountability Institute / Influence Map'
data_FF_v2.attrs['copyright'] = 'Copyright Climate Accountability Institute / Influence Map'
data_FF_v2.attrs['information'] = 'Emissions from carbon major entities'
data_FF_v2.attrs['details'] = 'Data provided by Richard Heede (Climate Accountability Institute) and Emmet Connaire (Influence Map) to Yann Quilcaille (ETH Zürich), treated by Yann Quilcaille for usage by Thomas Gasser (IIASA)'
data_FF_v2.attrs['distribution'] = 'This data is not for distribution'
data_FF_v2.attrs['version'] = '02 April 2024'

data_FF_v2.to_netcdf( '/landclim/yquilcaille/contributions_FF/FF/emissions_majors_1850-2022.nc', encoding={var: {"zlib": True} for var in data_FF_v2.variables} )

# comparison to former dataset
emissions_FF = xr.open_dataset( os.path.join(paths_in['FF'], 'emissions_majors_1850-2021.nc') )

fig = plt.figure( figsize=(20,10) )
plt.subplot(121)
plt.plot( np.arange(1850,2021+1), emissions_FF['emissions_CO2'].sel(year=slice(1850,2021)).sum('entity'), color='k', ls='--', lw=2, label='Version 2023' )
plt.plot( np.arange(1850,2022+1), data_FF_v2['emissions_CO2'].sel(year=slice(1850,2022)).sum('entity'), color='r', lw=2, label='Version 2024' )
plt.legend(loc=0)
plt.grid()
plt.xlim(1940,2022)
plt.ylabel('Emissions CO2 (MtCO2 / year)')
plt.subplot(122)
plt.plot( np.arange(1850,2021+1), emissions_FF['emissions_CH4'].sel(year=slice(1850,2021)).sum('entity'), color='k', ls='--', lw=2, label='Version 2023' )
plt.plot( np.arange(1850,2022+1), data_FF_v2['emissions_CH4'].sel(year=slice(1850,2022)).sum('entity'), color='r', lw=2, label='Version 2024' )
plt.legend(loc=0)
plt.grid()
plt.ylabel('Emissions CH4 (MtCH4 / year)')
plt.xlim(1940,2022)
fig.savefig('/landclim/yquilcaille/contributions_FF/FF/comparison_versions.png', dpi=300 )
