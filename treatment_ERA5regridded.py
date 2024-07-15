import os
import sys

import xarray as xr

from fcts_support_io import *

#==================================================================================================
# 0. OPTIONS
#==================================================================================================
# run in two times: first regridding using bash script on exo, then merge everything into a single file
# preparing paths
paths_in = {'ERA5':'/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_1h/processed/regrid_tmean1d',\
            'select_CMIP6':'/landclim/yquilcaille/contributions_FF/select_CMIP6',\
            'grid_cmip6ng':'/landclim/yquilcaille/contributions_FF/ancillary_data/grids/g025.txt'}
#==================================================================================================
#==================================================================================================



#==================================================================================================
# 1. REGRIDDING
#==================================================================================================
os.nice(19)
subindex_csl = int(sys.argv[1])
# listing files to do
files = [file for file in os.listdir( paths_in['ERA5'] ) if (file[-3:]=='.nc') and (file.split('.')[1]=='t2m')]

# identifying file for this csl
file = files[subindex_csl]
year = file.split('.')[4]

# regrid
regrid_cdo(path_file_in = os.path.join(paths_in['ERA5'], file),\
           path_file_out = os.path.join(paths_in['select_CMIP6'], 'ERA5-' + year + '_g025.nc'),\
           path_grid = paths_in['grid_cmip6ng'],\
           method = 'con2')
print('finished regridding on '+year)
#==================================================================================================
#==================================================================================================

    
#==================================================================================================
# 2. REGROUPING FILES
#==================================================================================================
# TO RUN AFTER ALL IS DONE.
if False:
    # getting regridded files
    files = [file for file in os.listdir( paths_in['select_CMIP6'] ) if (file[-3:]=='.nc') and (file[:5] == 'ERA5-')]
    
    # loading everything
    ds = xr.open_mfdataset( [os.path.join(paths_in['select_CMIP6'], file) for file in files], preprocess=preprocess_ERA5, combine='by_coords' )
    
    # saving into a single file
    ds.to_netcdf( os.path.join(paths_in['select_CMIP6'], 'ERA5_g025.nc'), encoding={var: {"zlib": True} for var in ds.variables} )
    print('PLEASE, DELETE intermediary regrided files AFTER CHECKING REGRIDDING, TO REDUCE STORAGE ON THE SERVER.')
#==================================================================================================
#==================================================================================================