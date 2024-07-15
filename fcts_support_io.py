import csv
import os
import unicodedata
import warnings

import cdo
import cftime as cft
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

CDO = cdo.Cdo()
from fcts_support_basic import *

warnings.simplefilter("ignore")



#---------------------------------------------------------------------------------------------------------------------------
# EVENTS
def func_prepare_emdat( path, disasters, start_year, end_year, option_detailed_prints=False ):
    if option_detailed_prints:
        print('Loading EM-DAT')
    
    # finding correct file:
    files = [file for file in os.listdir( path ) if file[-5:]=='.xlsx']
    
    # loading emdat
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emdat = pd.read_excel( os.path.join(path, files[0]) )

    # selecting names of columns
    names_columns = emdat.columns.values #emdat.iloc[5].values

    # cutting first rows
    #emdat = emdat.drop(axis=0, labels=np.arange(6))

    # renaming columns
    #emdat = emdat.rename( axis=1, mapper={c:names_columns[i_c] for i_c,c in enumerate(emdat.columns)} )

    # fixing the issue of the "'000US$'" that are 1000 $; and adding 2019 for adjusted according to EM-DAT website
    dic_tmp = {"Reconstruction Costs ('000 US$)":"Reconstruction Costs (US$)", \
               "Reconstruction Costs, Adjusted ('000 US$)":"Reconstruction Costs, Adjusted (US$)", \
               "Insured Damage ('000 US$)":"Insured Damage (US$)", \
               "Insured Damage, Adjusted ('000 US$)":"Insured Damage, Adjusted (US$)", \
               "Total Damage ('000 US$)":"Total Damage (US$)", \
               "Total Damage, Adjusted ('000 US$)":"Total Damage, Adjusted (2019 US$)" }
    for col in dic_tmp.keys():
        emdat[col] *= 1000
    emdat = emdat.rename( axis=1, mapper=dic_tmp )

    # transforming all of data into xarray to ease work
    emdat = emdat.to_xarray()

    # removing events not considered:
    if False:# checking organisation of types and subtypes
        tmp,a = {}, emdat['Disaster Subtype'].values
        for i, val in enumerate( emdat['Disaster Type'].values ):
            if val not in tmp:
                tmp[val] = []
            if a[i] not in tmp[val]:
                tmp[val].append( a[i] )
                
    # keeping required disasters:
    disasters = [clean_string(string) for string in disasters]
    list_selected, mapping_rejected = [], {}
    for i in emdat.index:
        clean_dis = clean_string( emdat['Disaster Type'].sel(index=i).values )
        clean_dissub = clean_string( emdat['Disaster Subtype'].sel(index=i).values )
        
        # checking whether to select this index, otherwise report it
        if (clean_dis in disasters) or (clean_dissub in disasters):
            list_selected.append( i )
        else:
            if clean_dis not in mapping_rejected:
                mapping_rejected[clean_dis] = []
            if clean_dissub not in mapping_rejected[clean_dis]:
                mapping_rejected[clean_dis].append( clean_dissub )
    if option_detailed_prints:
        print( 'Removing from EM-DAT the following disaster types: '+str( mapping_rejected ) )
    # emdat_removed_dissubtypes = emdat.sel(index = [i for i in emdat.index if i not in list_selected])
    emdat = emdat.sel(index = list_selected)

    # selecting only those we are interested in
    emdat_removed_years = emdat.where( (emdat['Start Year'] < start_year) | (emdat['Start Year'] > end_year), drop=True )
    emdat = emdat.where( (emdat['Start Year'] >= start_year) & (emdat['Start Year'] <= end_year), drop=True )
    return emdat, emdat_removed_years # emdat_removed_dissubtypes


def clean_string( string ):
    if string == np.nan:
        string = ''
    else:
        string = str(string)
        
    if len(string) > 0:
        # removing uppercases to recognize that
        string = string.lower()

        # removing useless spaces
        string = string.strip()

        # remove accents
        nfkd_form = unicodedata.normalize('NFKD', string)
        string = u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

        # remove some characters observed to cause issues
        for charac in np.arange(10):
            string = string.replace(str(charac),'')

        # replace some characters observed to cause issues
        for charac in ['-', '/', '!', '?']: # np \\
            string = string.replace(charac,' ')
    return string
#---------------------------------------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------------------------------------
# GEOGRAPHICAL BOUNDARIES
def func_prepare_geobounds( path, source='gadm', option_detailed_prints=False ):
    if option_detailed_prints:
        print('Loading geo bounds: '+source)
        
    if source == 'WAB':
        # finding correct file:
        files = [file for file in os.listdir( path ) if file[-4:]=='.csv']

        # read file
        with open(os.path.join(path, files[0]), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar='|')
            dict_output = {}
            for row in reader:
                # creating dictionary using header
                if len(dict_output) == 0:
                    header = row
                    for item in row:
                        dict_output[item] = []

                # adding the new line to the dictionary
                else:
                    for i, item in enumerate(row):
                        dict_output[header[i]].append( item )
                        
        return dict_output
    
    elif source == 'gadm':
        # finding correct file:
        files = [file for file in os.listdir( path ) if file[-5:]=='.gpkg']
        
        # loading
        geobounds = gpd.read_file(os.path.join( path, files[0] ) )#, layer='countries')
        
        # preparing dictionary of ISO --> indexes
        dict_geobounds = {}
        for i in range(len(geobounds)):
            if geobounds.iloc[i]['GID_0'] not in dict_geobounds:
                dict_geobounds[geobounds.iloc[i]['GID_0']] = []
            dict_geobounds[geobounds.iloc[i]['GID_0']].append( i )
        if option_detailed_prints:
            print('Finished loading geographical database')
        return geobounds, dict_geobounds
    
    else:
        raise Exception("Unknown source.")
#---------------------------------------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------------------------------------
# OBSERVATIONS
def preprocess_BEST(ds):
    # need to rename axis
    ds = ds.rename( {'longitude':'lon', 'latitude':'lat'} )
    
    # need to adapt the time axis
    if False:
        tmp_time = ds['date_number'].values
        
    else:
        # preparing time values
        year = np.array( np.array(ds.year.values,dtype=int), dtype=str)
        month = np.array( np.array( ds.month.values,dtype=int), dtype=str)
        day = np.array( np.array( ds.day.values,dtype=int), dtype=str)
        for val in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
            month[np.where( month==val )] = '0'+val
            day[np.where( day==val )] = '0'+val

        # creating new time axis
        tmp_time = [np.datetime64('-'.join( val ), 'D') for val in zip(year, month, day)]
        
    # climato
    ds['temperature'].attrs['climatology'] = ds['climatology'].attrs['long_name']

    # removing some variables, some using 'time' coordinate
    ds = ds.drop( ['year', 'month', 'day_of_year', 'day'] )# 'climatology'
    
    # improving coordinate
    ds.coords['time'] = tmp_time

    return ds
    
    
def func_prepare_BEST( path, path_gmt=None, sea_ice_temp_from='air_temperatures', list_vars=['tavg'], option_detailed_prints=False ):
    # finding files
    files = [file for file in os.listdir( path ) if file[-3:]=='.nc']

    if list_vars is None:
        list_vars = set( [file.split('_')[1].lower() for file in files] )

    # preparing
    ds = {}

    # loading
    for var in list_vars:
        if option_detailed_prints:
            print('Loading observations: '+var)

        # list files for this specific variable
        tmp = [file for file in files if file.split('_')[1].lower() == var]

        # loading files
        ds[var] = xr.open_mfdataset( [os.path.join(path, file) for file in tmp], preprocess=preprocess_BEST, combine='by_coords'  )

        # add climato
        tmp = ds[var]['temperature'].compute()
        tmp_c = ds[var]['climatology'].isel(time=0).drop('time').compute()
        for i, t in enumerate( tmp.time ):
            if option_detailed_prints:
                print( 'Re-adding climatology of BEST: '+str(np.round(100*(i+1)/tmp.time.size,1))+'%', end='\r' )
            if (t.dt.is_leap_year.values) and t.dt.dayofyear == 31 + 29:
                leaped = np.array([1,0])
            elif (t.dt.is_leap_year.values) and t.dt.dayofyear > 31 + 29:
                leaped = 1
            else:
                leaped = 0
            clim_day = tmp_c.sel(day_number = t.dt.dayofyear.values - 1 - leaped)
            if 'day_number' in clim_day.dims:
                clim_day = clim_day.mean('day_number')
            tmp.loc[{'time':t}] += clim_day

        # will return that
        ds[var]['temperature'] = tmp

        # selecting only on average
        if (var == 'tavg') and (path_gmt is not None):
            gmt = xr.open_dataset( os.path.join(path_gmt,'BEST_GMT.nc') )
            gmt_best = gmt['temperature'].sel( sea_ice_temp_from=sea_ice_temp_from ).drop( 'sea_ice_temp_from' )
            
            # smoothing following WWA approach: value at t is average over this year + 3 last years
            val_gmt = [gmt_best.sel(time=slice(yr-3,yr)).mean('time').assign_coords(time=yr) for yr in gmt_best.time.values]
            gmt_best = xr.concat( val_gmt, dim='time' )
    
    if path_gmt is not None:
        return ds, gmt_best
    else:
        return ds


def func_prepare_ERA5( path, path_gmt=None, list_vars=['t2m'], option_detailed_prints=False ):
    # finding files
    files = [file for file in os.listdir( path ) if (file[-3:]=='.nc')]

    if list_vars is None:
        list_vars = set( [file.split('.')[1].lower() for file in files] )

    # preparing
    ds = {}

    # loading
    for var in list_vars:
        if option_detailed_prints:
            print('Loading observations: '+var)

        # list files for this specific variable
        tmp = [file for file in files if file.split('.')[1].lower() == var]

        # loading files
        ds[var] = xr.open_mfdataset( [os.path.join(path, file) for file in tmp], preprocess=preprocess_ERA5, combine='by_coords' )

        # selecting only average
        if (var == 't2m') and (path_gmt is not None):
            gmt = xr.open_dataset( os.path.join(path_gmt,'ERA5_GMT.nc') )
            gmt_data = gmt['t2m']
            
            # smoothing following WWA approach: value at t is average over this year + 3 last years
            val_gmt = [gmt_data.sel(time=slice(yr-3,yr)).mean('time').assign_coords(time=yr) for yr in gmt_data.time.values]
            gmt_data = xr.concat( val_gmt, dim='time' )
            
    if path_gmt is not None:
        return ds, gmt_data
    else:
        return ds



def preprocess_ERA5(ds):
    # need to rename variables
    #ds = ds.rename( {'t2m':'temperature'} )
    
    # removing some variables, some using 'time' coordinate
    ds = ds.drop( ['time_bnds'] )
    
    # kelvin to celsius
    if 't2m' in ds.variables:
        ds['t2m'] -= 273.15
        ds['t2m'].attrs['units'] = 'degC'
    
    # improving time coordinate for comparison to BEST, no impact on selection
    ds.coords['time'] = [np.datetime64( str(t).split('T')[0]+'T' + '00' + ':' + '00' + ':' + '00' ) for t in ds.time.values]
    return ds
#---------------------------------------------------------------------------------------------------------------------------









#---------------------------------------------------------------------------------------------------------------------------
# CMIP6-ng DATA (adapted from my personal pipeline for annual indicators)
class files_cmip6():
    """
        This class gathers all required files for use in the pipeline for annual indicators.
    """
    
    
    #----------------------------------------------------------------------
    # INITALIZATION
    def __init__(self, path_cmip6, path_exclusions_cmip6, var_input, list_experiments, list_members, option_detailed_prints=False):
        """
            Initialization of the class:
            
            path_cmip6: str
                Path where all CMIP6 files are
                
            var_input: str
                Name of the CMIP6 variable as input.
                
            list_experiments: list
                List of strings, each string being a CMIP6 experiment.
        """
        self.path_cmip6 = path_cmip6
        self.var_input = var_input
        self.list_experiments = list_experiments
        self.list_members = list_members
        self.option_detailed_prints = option_detailed_prints
        
        # directly reading exceptions for this variable: some whole run
        csv_exclusion = os.path.join(path_exclusions_cmip6, 'exclusions_'+var_input+'.csv')
        if os.path.isfile( csv_exclusion )==False:
            # no exceptions created yet, adding file if need to add some
            with open(csv_exclusion, 'w', newline='') as csvfile:
                writerempty = csv.writer(csvfile, delimiter=' ')
            self.exclusions = []
        else:
            with open( csv_exclusion, newline='') as csvfile:
                read = csv.reader(csvfile, delimiter=',')
                self.exclusions = [row[:4+1] for row in read]
            
        # directly reading exceptions for this variable: some individual files
        csv_exclusion_files = os.path.join(path_exclusions_cmip6, 'exclusions_'+var_input+'_files.csv')
        if os.path.isfile( csv_exclusion_files )==False:
            # no exceptions created yet, adding file if need to add some
            with open(csv_exclusion_files, 'w', newline='') as csvfile:
                writerempty = csv.writer(csvfile, delimiter=' ')
            self.exclusions_files = []
        else:
            with open( csv_exclusion_files, newline='') as csvfile:
                read = csv.reader(csvfile, delimiter=',')
                self.exclusions_files = [row[0] for row in read]
    #----------------------------------------------------------------------
        
        
        
    #----------------------------------------------------------------------
    # GATHERING ALL FILES
    def gather_files(self, forced_domain='day', option_single_grid=True):
        """
            Produces the list of paths for all the required & available files.
            
            forced_domain: None or str
                If a str is prescribed, this is the domain where the input variable is meant to be. If nothing is prescribed, will look for it. Though, observed multiple definitions. "day" may be a better default.
                
            option_single_grid: boolean
                Some ESMs provided runs under different grids. Because the outputs will be regridded onto a single grid, may want to choose only the first one (=True). Otherwise, will run them all (=False).            
        """
        # Checking for the domain
        #self.domains = self.find_domain()
        self.domains = forced_domain
            
        # Deduce corresponding time resolution
        self.time_res = [ self.find_timeres(d) for d in self.domains ]
        
        # Building nested dictionary for files
        self.nested_dict_files = self.build_nested(option_single_grid=option_single_grid)
        
        # Building list of files
        self.dico_files_full = {}
        for dom in self.domains:# first, doing mon domain
            for xp in self.nested_dict_files[dom]:
                for esm in self.nested_dict_files[dom][xp]:
                    for memb in self.nested_dict_files[dom][xp][esm]:
                        for grid in self.nested_dict_files[dom][xp][esm][memb]:
                            # if "..mon" domain in here, this one precedes over "day" domain. If not, will get just this one domain.
                            if ' / '.join([ self.domains[0], xp, esm, memb]) not in self.dico_files_full:
                                path = os.path.join(self.path_cmip6, xp, dom, self.var_input, esm, memb, grid)
                                files = [fl for fl in os.listdir(path) if '.nc' in fl]
                                if len(files) > 0:
                                    files.sort()
                                    # writing here the excluded files: did not find a better solution so far
                                    self.dico_files_full[ ' / '.join([dom, xp, esm, memb]) ] = [os.path.join(path,fl) for fl in files if os.path.join(path,fl) not in self.exclusions_files]
        self.n_runs_full = len(self.dico_files_full)
    
    @staticmethod
    def find_timeres(dom):
        if dom in ['3hr', 'CF3hr', 'E3hr']:
            time_res = '3hr'
            
        elif dom in ['AERday', 'CFday', 'day', 'Oday', 'SIday']:
            time_res = 'day'
           
        elif dom in ['AERmon', 'Amon', 'Emon', 'LImon', 'Lmon', 'Omon', 'SImon']:
            time_res = 'mon'
           
        elif dom in ['fx', 'Ofx']:
            time_res = 'fx'
           
        else:
            raise Exception("This domain has not been implemented.")
        
        return time_res

        
    def build_nested(self, option_single_grid):
        """
            Produces a nested dictionnary of available files to generate all paths.
            
            option_single_grid: boolean
                Some ESMs provided runs under different grids. Because the outputs will be regridded onto a single grid, may want to choose only the first one (=True). Otherwise, will run them all (=False).
        """
        # preparing list of repositories with multiple grids:
        self.several_grids = []
    
        # defining level 0 of nested dictionary
        nested_dict_files = {}
        
        # defining level 1 of nested dictionary: domains
        nested_dict_files = {d:{} for d in self.domains}
        for dom in self.domains:
            
            # defining level 2 of nested dictionary: experiments
            nested_dict_files[dom] = {xp:{} for xp in self.list_experiments}
            for xp in self.list_experiments:
                nested_dict_files[dom][xp] = {}
                path_esm = os.path.join(self.path_cmip6, xp, dom, self.var_input)
                if os.path.isdir( path_esm ):
                    esms = os.listdir( path_esm )

                    # defining level 3 of nested dictionary: esms
                    nested_dict_files[dom][xp] = {esm:{} for esm in esms}
                    for esm in esms:
                        members = os.listdir( os.path.join(path_esm, esm) )
                        
                        # keeping only required members
                        members = [memb for memb in members if memb in self.list_members]

                        # defining level 4 of nested dictionary: members
                        nested_dict_files[dom][xp][esm] = {memb:{} for memb in members}
                        for memb in members:
                            grids = os.listdir( os.path.join(path_esm, esm, memb) )

                            # removing here exclusions through the grids
                            grids_to_remove = [ grid for grid in grids if [dom, esm, xp, memb, grid] in self.exclusions ]
                            for grid in grids_to_remove:
                                grids.remove(grid)

                            if option_single_grid and len(grids)>1:
                                self.several_grids.append( [xp,esm,memb,grids] )
                                grids = [grids[0]]

                            # defining level 5 of nested dictionary: grids
                            nested_dict_files[dom][xp][esm][memb] = grids
                    
            # may need to remove further this ensemble if removed grids on level 4 because of exclusions, but it was the only one
            for xp in list(nested_dict_files[dom]):
                for esm in list(nested_dict_files[dom][xp]):
                    for memb in list(nested_dict_files[dom][xp][esm]):
                        # checking if by removing exclusions, removed the only feasible grid: then remove this member
                        if len(nested_dict_files[dom][xp][esm][memb]) == 0:
                            del nested_dict_files[dom][xp][esm][memb]
                    # checking if by removing exclusions, removed the only feasible members: then remove this esm
                    if len(nested_dict_files[dom][xp][esm]) == 0:
                        del nested_dict_files[dom][xp][esm]
                # checking if by removing exclusions, removed the only feasible esms: then remove this xp
                if len(nested_dict_files[dom][xp]) == 0:
                    del nested_dict_files[dom][xp]                    

        return nested_dict_files
    #----------------------------------------------------------------------


    
    
    #----------------------------------------------------------------------
    # FILTERING FILES
    def filter_runs(self):        
        # building a list of runs that will have to be used here
        self.dico_runs_filtered = self.build_list_runs()
        self.n_runs_filtered = sum([len(self.dico_runs_filtered[run]) for run in self.dico_runs_filtered])
        
        # building the list of files that have to be loaded
        self.dico_files_filtered = {}
        for run in self.dico_runs_filtered:
            xps = self.dico_runs_filtered[run]
            dom, esm, memb = str.split( run, ' / ' )
            items = [' / '.join([dom, xp, esm, memb]) for xp in xps]
            self.dico_files_filtered[run] = {it:self.dico_files_full[it] for it in items}
            
        self.filter_files()
        
        # to facilitate loops, directly creating this list
        self.list_esms = list(set([ str.split(item, ' / ')[1] for item in self.dico_files_filtered.keys() if len(self.dico_files_filtered[item])==len(self.list_experiments) ]))
        self.list_esms.sort()
        
        # to facilitate loading files, creating this new dictionary
        self.dico_esms_files = {esm:[] for esm in self.list_esms}
        for day_esm_memb in self.dico_files_filtered.keys():
            esm = str.split( day_esm_memb, ' / ' )[1]
            if len(self.dico_files_filtered[day_esm_memb]) == len(self.list_experiments):
                for day_esm_memb_scen in self.dico_files_filtered[day_esm_memb]:
                    self.dico_esms_files[esm] += self.dico_files_filtered[day_esm_memb][day_esm_memb_scen]
        for esm in self.list_esms:
            self.dico_esms_files[esm].sort()


    def build_list_runs(self):
        # creating a dictionary that gathers the available scenarios for each esm / memb
        dico_tmp = {}
        for run in self.dico_files_full.keys():
            dom, xp, esm, memb = str.split( run, ' / ' )
            id_run = ' / '.join([dom, esm, memb])
            if id_run not in dico_tmp.keys():
                dico_tmp[id_run] = []
            dico_tmp[id_run].append( xp )
            
        # making a quick check: normally, the reference period must be within the historical: this experiment should be there.
        self.runs_nohistorical = []
        for id_run in dico_tmp:
            if 'historical' not in dico_tmp[id_run]:
                self.runs_nohistorical.append(id_run)
        if ~self.option_detailed_prints:
            print("Missing the experiment 'historical' for "+str(len(self.runs_nohistorical))+" ESM x member, will not calculate its other experiments: consult 'runs_nohistorical' for the details of those removed.")
        for id_run in self.runs_nohistorical:
            del dico_tmp[id_run]
                
        # aggregating together experiments of the same esm / memb, because some indicators relie on the historical period as a reference
        list_tmp = []
        for run in dico_tmp.keys():
            dom, esm, memb = str.split( run, ' / ' )

            # checking if the regridded files are computed for all the experiments of this esm / memb
            test_needrun = {}
            for xp in dico_tmp[run]:
                test_needrun[xp] = True

            if np.any( list(test_needrun.values()) ):# if so, adding them all: required because of historical period as reference for scenarios
                list_tmp.append( run )

        # reduces the list of runs to use
        return {run:dico_tmp[run] for run in list_tmp}    
    
    def filter_files(self):
        # Some files may be duplicated (eg 201501-206412; 201501-210012; 206501-210012): filtering this case
        # Also filtering those that are too early
        self.all_removed = []
        tmp = list(self.dico_files_filtered.keys())
        for run in self.dico_files_filtered:
            for run_xp in self.dico_files_filtered[run]:
                files = self.dico_files_filtered[run][run_xp]
                dom, xp, esm, memb = str.split( run_xp, ' / ' )
                res = self.time_res[ self.domains.index(dom) ]
                path = os.path.join(self.path_cmip6, xp, dom, self.var_input, esm, memb)

                if len(files) >= 1:
                    xp = run_xp.split(' / ')[1]
                    
                    # getting dates of each files
                    times_files = {}
                    for file_name in files:
                        timestamp = os.path.splitext(os.path.basename(file_name))[0].split('_')[-1].split('-')
                        if res == 'mon':
                            times_files[file_name] = [[int(timestamp[0][:4]), int(timestamp[0][4:6])], [int(timestamp[1][:4]), int(timestamp[1][4:6])]]
                        elif res == 'day':
                            times_files[file_name] = [[int(timestamp[0][:4]), int(timestamp[0][4:6]), int(timestamp[0][6:8])], [int(timestamp[1][:4]), int(timestamp[1][4:6]), int(timestamp[1][6:8])]]

                    # files that will be removed:
                    to_remove = []
                    
                    # look for which files start at a correct year: ome scenarios start before 2015
                    i_start = 0
                    if xp in ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']:
                        while (i_start < len(files)) and (times_files[files[i_start]][0][0] != 2015):
                            to_remove.append( files[i_start] )
                            i_start += 1
                        if i_start == len(files):
                            raise Exception("No files after 2014 for this scenario? Check there: "+path)
                    elif xp in ['ssp534-over']:
                        while (i_start < len(files)) and (times_files[files[i_start]][0][0] < 2015):
                            to_remove.append( files[i_start] )
                            i_start += 1
                        if i_start == len(files):
                            raise Exception("No files after 2014 for this scenario? Check there: "+path)
                    elif xp in ['historical']:
                        while (i_start < len(files)) and (times_files[files[i_start]][0][0] != 1850):
                            to_remove.append( files[i_start] )
                            i_start += 1
                        if i_start == len(files):
                            raise Exception("No files starting in 1850 for this run? Check there: "+path)
                    # checking if the dates work together:
                    file_date = [files[i_start], times_files[files[i_start]]] # first file
                    for file_name in files[i_start+1:]:
                        if times_files[file_name][0] == file_date[1][0]: # same file?
                            if times_files[file_name][1] == file_date[1][1]:
                                raise Exception("Two identical files? "+path)
                            elif times_files[file_name][1][0] < file_date[1][1][0]: # keep the first one, cf years. remove the current one.
                                to_remove.append(file_name)
                            elif times_files[file_name][1][0] > file_date[1][1][0]: # keep the other one, cf years. remove the first one.
                                to_remove.append(file_date[0])
                                file_date = [file_name, times_files[file_name]]
                            else:
                                raise Exception("Some strange case here: "+path)
                                
                        elif times_files[file_name][0][0] == file_date[1][1][0] + 1:# next year, ok
                            if (times_files[file_name][0][1] != 1)  or  ((res=='day') and (times_files[file_name][0][2] != 1)):
                                # failed control of the month and eventually of the day
                                raise Exception("Observing a jump in timestamps, missing data? "+path)
                                
                            elif (xp in ['historical']) and (times_files[file_name][0][0] > 2014):
                                # failed control of the year: extended the scenario, cutting there
                                to_remove.append(file_name)
                                
                            else:
                                file_date = [file_name, times_files[file_name]]
                                
                        else:
                            # not the next year!
                            to_remove.append(file_name)
                            
                    # checking if reached at least 2100 for scenarios
                    if (xp in ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585', 'ssp534-over'])  and  (file_date[1][1][0] < 2100) or (xp in ['historical'])  and  (file_date[1][1][0] < 2014):
                        raise Exception('Missing years for '+path+', finished at '+str(file_date[1][1][0]))
                        
                    for file_remove in to_remove:
                        self.dico_files_filtered[run][run_xp].remove( file_remove )
                        self.all_removed.append( file_remove )
    #----------------------------------------------------------------------
                        
                        
    #----------------------------------------------------------------------
    def load_all(self, path_gmt=None):
        # loading files
        self.data_esm, self.gmt_esm = {}, {}
        for esm in self.list_esms:
            if ~self.option_detailed_prints:
                print('preparing '+esm)
            self.data_esm[esm] = xr.open_mfdataset( self.dico_esms_files[esm], preprocess=self.preprocess_cmip6 )
            
            # removing some redundant info in this specific case
            self.data_esm[esm] = self.data_esm[esm].sel(esm=esm).drop('esm')
            
            if path_gmt is not None:
                # load GMT of the ESM
                self.gmt_esm[esm] = xr.open_dataset( os.path.join(path_gmt, esm+'_GMT.nc') )

                # checking that correct conditions
                if ('tas' not in self.gmt_esm[esm]) or \
                (self.gmt_esm[esm].member.values != self.list_members) or \
                (self.gmt_esm[esm]['tas'].attrs['experiments'] != str(self.list_experiments)):
                    raise Exception('Wrong GMT calculation? Check that.')

                # Droping information no longer useful
                if 'esm' in self.gmt_esm[esm].coords:
                    self.gmt_esm[esm] = self.gmt_esm[esm]['tas'].sel(esm=esm).drop('esm')
                if len(self.list_members) == 1:
                    self.gmt_esm[esm] = self.gmt_esm[esm].sel(member=self.list_members)#.drop('member')

                # smoothing following WWA approach: value at t is average over this year + 3 last years
                # preparing shifted time serie
                if type( self.gmt_esm[esm].time.values[0] ) == np.datetime64:
                    yr_0 = pd.DatetimeIndex( self.gmt_esm[esm].time.values ).shift(periods=-3, freq='Y')
                else:
                    yr_0 = xr.CFTimeIndex( self.gmt_esm[esm].time.values ).shift( freq='Y', n=-3 )
                # averaging
                val_gmt = []
                for i, yr in enumerate( self.gmt_esm[esm].time.values ):
                    val_gmt.append( self.gmt_esm[esm].sel(time=slice(yr_0[i],yr)).mean('time').assign_coords(time=yr) )
                self.gmt_esm[esm] = xr.concat( val_gmt, dim='time' )

                # coords time to change into years
                if type( self.data_esm[esm].time.values[0] ) == np.datetime64:
                    self.gmt_esm[esm].coords['time'] = np.array( pd.DatetimeIndex(self.gmt_esm[esm].time).year )
                else:
                    self.gmt_esm[esm].coords['time'] = np.array( [val.year for val in self.gmt_esm[esm].time.values] )
                # NOT CLOSING to keep them and load them when required
    
    @staticmethod
    def preprocess_cmip6(ds):
        path = ds.encoding['source']

        if 'GFDL-CM4' in path and 'historical' in path:
            ds = ds.drop('height', errors='ignore')
        if 'EC-Earth3' in path:
            ds = ds.assign_coords(lat=np.around(ds['lat'], 2))

        # adding this attribute to gather multiple experiments
        ds = ds.expand_dims( {"member":[ds.attrs["variant_label"]]}) # "experiment_id": "scen", "esm":[ds.attrs["source_id"]]
        for var in ["variant_label"]:
            del ds.attrs[var]

        # dropping some useless & problematic variables
        ds = ds.drop( [var for var in ['time_bnds', 'lat_bnds', 'lon_bnds', 'time_bounds', 'lat_bounds', 'lon_bounds', 'height'] if var in list(ds.keys())+list(ds.coords.keys())] )

        # kelvin to celsius
        if 'tas' in ds:
            if ds['tas'].attrs['units'] == 'K':
                ds['tas'] -= 273.15
                ds['tas'].attrs['units'] = 'degC'
        
        if False:
            # dealing with time axis that are too long (e.g. tas_day_CESM2_historical_r4i1p1f1_gn_20100101-20150103.nc)
            if ds.attrs['experiment_id'] in ['historical']:
                # start of file
                start = ds.time.values[0]
                format_s = type( start )
                # normal end of file
                if format_s == np.datetime64:
                    s2 = pd.DatetimeIndex([start])[0]
                    tmp = {'hr':str(s2.hour), 'mi':str(s2.minute), 'se':str(s2.second)}
                    for kk in tmp:
                        if len(tmp[kk]) == 1:
                            tmp[kk] = '0' + tmp[kk]
                    end = format_s( '2014-12-31T' + tmp['hr'] + ':' + tmp['mi'] + ':' + tmp['se'])                
                else:
                    end = format_s( year=2014, month=12, day=31, hour=0, minute=0, second=0, microsecond=0)
                # cutting
                ds = ds.sel(time = slice(start, end))
                
        return ds
    
    @staticmethod
    def format_date( dict_date, val0=None ):

        # creating date
        if (type(val0) == np.datetime64) or (val0 is None):
            # preparing date
            dic_dt = {}
            for dt in ['Year', 'Month', 'Day']:
                dic_dt[dt] = str(int(dict_date[dt]))
                if len(dic_dt[dt]) == 1:
                    dic_dt[dt] = '0'+dic_dt[dt]
            return np.datetime64( '-'.join( [dic_dt['Year'], dic_dt['Month'], dic_dt['Day']] ), 'D')
        else:
            format_t = type(val0)
            return format_t( year=dict_date['Year'], month=dict_date['Month'], day=dict_date['Day'], hour=val0.hour, minute=val0.minute, second=val0.second, microsecond=val0.microsecond)
    #----------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------









#---------------------------------------------------------------------------------------------------------------------------
# CMIP6 DATA (from my personal pipeline for annual indicators)
class files_cmip6ng():
    """
        This class gathers all required files for use in the pipeline for annual indicators.
    """
    
    
    #----------------------------------------------------------------------
    # INITALIZATION
    def __init__(self, path_cmip6ng, var_input, list_experiments, list_members, domain='day', option_detailed_prints=False):
        """
            Initialization of the class:
            
            path_cmip6: str
                Path where all CMIP6 files are
                
            var_input: str
                Name of the CMIP6 variable as input.
                
            list_experiments: list
                List of strings, each string being a CMIP6 experiment.
        """
        self.path_cmip6ng = path_cmip6ng
        self.var_input = var_input
        self.list_experiments = list_experiments
        self.list_members = list_members
        self.domain = 'day'
        self.option_detailed_prints = option_detailed_prints
        
        # gather files
        self.gather_files()
    #----------------------------------------------------------------------
        
        
        
    #----------------------------------------------------------------------
    # GATHERING ALL FILES
    def gather_files(self):
        """
            Produces the list of paths for all the required & available files.
            
            forced_domain: None or str
                If a str is prescribed, this is the domain where the input variable is meant to be. If nothing is prescribed, will look for it. Though, observed multiple definitions. "day" may be a better default.
                
            option_single_grid: boolean
                Some ESMs provided runs under different grids. Because the outputs will be regridded onto a single grid, may want to choose only the first one (=True). Otherwise, will run them all (=False).            
        """
        # Checking for files
        full_path = os.path.join(self.path_cmip6ng, self.var_input, self.domain, 'g025')
        list_files = os.listdir( full_path )
        
        # selecting those matching criterias
        self.dico_files = {}
        for file in list_files:
            _, _, esm, scen, member, _ = str.split( file, '_')
            
            # verifying whether keeps it or not
            kept = True
            # is it in required experiments?
            if (self.list_experiments is not None) and (scen not in self.list_experiments):
                kept = False
            # is it in required members?
            if (self.list_members is not None) and (member not in self.list_members):
                kept = False
                
            # accepting the run
            if kept:
                if esm not in self.dico_files:
                    self.dico_files[esm] = {}
                if member not in self.dico_files[esm]:
                    self.dico_files[esm][member] = {}
                if scen not in self.dico_files[esm][member]:
                    self.dico_files[esm][member][scen] = os.path.join( full_path, file )
        
        # checks
        esms_to_remove = []
        for esm in self.dico_files:
            # checking that each ESM has same members for each experiment
            members_to_remove = []
            for member in self.dico_files[esm]:
                if len( self.dico_files[esm][member] ) < len(self.list_experiments):
                    # not all experiments in there, removing this member after loop
                    members_to_remove.append( member )
            for member in members_to_remove:
                del self.dico_files[esm][member]
                
            # checking that each ESM has at least 1 member, removing after loop
            if len( self.dico_files[esm] ) == 0:
                esms_to_remove.append( esm )
        for esm in esms_to_remove:
            del self.dico_files[esm]
            
        # creating the final list for loading
        self.list_files_kept = {}
        tmp = list(self.dico_files.keys())
        tmp.sort()
        for esm in tmp:
            if esm not in self.list_files_kept:
                self.list_files_kept[esm] = []
            for member in self.list_members:
                for scen in self.list_experiments:
                    self.list_files_kept[esm].append( self.dico_files[esm][member][scen] )
    #----------------------------------------------------------------------
                        
                        
    #----------------------------------------------------------------------
    def load_all(self, path_gmt=None):
        self.data_esm, self.gmt_esm = {}, {}
        for esm in self.list_files_kept:
            if self.option_detailed_prints:
                print( 'Loading '+esm)
            # loading files
            self.data_esm[esm] = xr.open_mfdataset( self.list_files_kept[esm], preprocess=self.preprocess_cmip6ng )
            
            # removing some redundant info in this specific case
            self.data_esm[esm] = self.data_esm[esm].sel(esm=esm).drop('esm')
            
            if path_gmt is not None:
                # load GMT of the ESM
                self.gmt_esm[esm] = xr.open_dataset( os.path.join(path_gmt, esm+'_GMT.nc') )

                # checking that correct conditions
                if ('tas' not in self.gmt_esm[esm]) or \
                (self.gmt_esm[esm].member.values != self.list_members) or \
                (self.gmt_esm[esm]['tas'].attrs['experiments'] != str(self.list_experiments)):
                    raise Exception('Wrong GMT calculation? Check that.')

                # Droping information no longer useful
                if 'esm' in self.gmt_esm[esm].coords:
                    self.gmt_esm[esm] = self.gmt_esm[esm]['tas'].sel(esm=esm).drop('esm')
                if len(self.list_members) == 1:
                    self.gmt_esm[esm] = self.gmt_esm[esm].sel(member=self.list_members)#.drop('member')

                # smoothing following WWA approach: value at t is average over this year + 3 last years
                # preparing shifted time serie
                if type( self.gmt_esm[esm].time.values[0] ) == np.datetime64:
                    yr_0 = pd.DatetimeIndex( self.gmt_esm[esm].time.values ).shift(periods=-3, freq='Y')
                else:
                    yr_0 = xr.CFTimeIndex( self.gmt_esm[esm].time.values ).shift( freq='Y', n=-3 )
                # averaging
                val_gmt = []
                for i, yr in enumerate( self.gmt_esm[esm].time.values ):
                    val_gmt.append( self.gmt_esm[esm].sel(time=slice(yr_0[i],yr)).mean('time').assign_coords(time=yr) )
                self.gmt_esm[esm] = xr.concat( val_gmt, dim='time' )

                # coords time to change into years
                if type( self.data_esm[esm].time.values[0] ) == np.datetime64:
                    self.gmt_esm[esm].coords['time'] = np.array( pd.DatetimeIndex(self.gmt_esm[esm].time).year )
                else:
                    self.gmt_esm[esm].coords['time'] = np.array( [val.year for val in self.gmt_esm[esm].time.values] )
                # NOT CLOSING to keep them and load them when required

    @staticmethod
    def preprocess_cmip6ng(ds):
        path = ds.encoding['source']
        
        if 'height' in ds:
            ds = ds.drop('height', errors='ignore')
        ds = ds.drop('file_qf', errors='ignore')

        # adding this attribute to gather multiple experiments --> not scenario, to directly stuck them together
        ds = ds.expand_dims( {"member":[ds.attrs["variant_label"]], "esm":[ds.attrs["source_id"]]} )#, "scen": [ds.attrs["experiment_id"]]
        for var in ["variant_label"]:
            del ds.attrs[var]

        # dropping some useless & problematic variables
        ds = ds.drop( [var for var in ['time_bnds', 'lat_bnds', 'lon_bnds', 'time_bounds', 'lat_bounds', 'lon_bounds', 'height'] if var in list(ds.keys())+list(ds.coords.keys())] )

        # kelvin to celsius
        if 'tas' in ds:
            if ds['tas'].attrs['units'] == 'K':
                ds['tas'] -= 273.15
                ds['tas'].attrs['units'] = 'degC'
                
        return ds
    
    @staticmethod
    def format_date( dict_date, val0=None ):

        # creating date
        if (type(val0) == np.datetime64) or (val0 is None):
            # preparing date
            dic_dt = {}
            for dt in ['Year', 'Month', 'Day']:
                dic_dt[dt] = str(int(dict_date[dt]))
                if len(dic_dt[dt]) == 1:
                    dic_dt[dt] = '0'+dic_dt[dt]
            return np.datetime64( '-'.join( [dic_dt['Year'], dic_dt['Month'], dic_dt['Day']] ), 'D')
        else:
            format_t = type(val0)
            if format_t in [cft._cftime.Datetime360Day] and dict_date['Day'] == 31:
                return format_t( year=dict_date['Year'], month=dict_date['Month'], day=30, hour=val0.hour, minute=val0.minute, second=val0.second, microsecond=val0.microsecond)
            else:
                return format_t( year=dict_date['Year'], month=dict_date['Month'], day=dict_date['Day'], hour=val0.hour, minute=val0.minute, second=val0.second, microsecond=val0.microsecond)
    #----------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------







#---------------------------------------------------------------------------------------------------------------------------
# OSCAR DATA
def func_prepare_OSCAR_GMT( path_gmt, option_detailed_prints=False ):
    # loading OSCAR data
    OSCAR_ctrl = xr.open_dataset( os.path.join(path_gmt,'Out_ctrl_to_share.nc') )
    OSCAR_minus = xr.open_dataset( os.path.join(path_gmt,'Out_ctrl_minus_majors_to_share.nc') )
    
    # Preparing output
    OUT = xr.Dataset()
    OUT['dGMT_entities_values'] = OSCAR_ctrl['D_Tg'] - OSCAR_minus['D_Tg']
    OUT['config_weights'] = OSCAR_ctrl.w
    OUT['dGMT_entities_mean'] = OUT['dGMT_entities_values'].weighted(OSCAR_ctrl.w).mean(dim=("config"))
    OUT['dGMT_entities_std'] = OUT['dGMT_entities_values'].weighted(OSCAR_ctrl.w).std(dim=("config"))
    
    percentiles = np.array([2.5, 50, 97.5])
    OUT.coords['percentiles'] = percentiles
    OUT['dGMT_entities_percentiles'] = xr.zeros_like( OUT['percentiles'] * OUT['dGMT_entities_mean'] )
    for i, entity in enumerate(OUT.entity):
        if option_detailed_prints:
            print('Percentiles of contributions in GMT for entity '+str(i+1)+'/'+str(OUT.entity.size), end='\r')
        for yr in OUT.year:
            OUT['dGMT_entities_percentiles'].loc[{'entity':entity,'year':yr}] = weighted_quantile(OUT['dGMT_entities_values'].sel(entity=entity,year=yr).values,\
                                                                                                    percentiles/100, sample_weight=OSCAR_ctrl.w.values)
    
    # just renaming the time axis
    OUT = OUT.rename({'year':'time'})
    if False:
        tmp = OSCAR_ctrl['D_Tg'] - OSCAR_ctrl['D_Tg'].sel(year=slice(1850,1900)).mean('year')
        OUT_ctrl = xr.Dataset()
        OUT_ctrl['GMT'] = tmp.weighted(OSCAR_ctrl.w).mean(dim=("config"))
        OUT_ctrl['GMT_std'] = tmp.weighted(OSCAR_ctrl.w).std(dim=("config"))
        OUT_ctrl = OUT_ctrl.rename({'year':'time'})
        return OUT, OUT_ctrl
    else:
        return OUT
#---------------------------------------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------------------------------------
# Global Carbon Budget
def func_prepare_GCB( path_gcb ):
    # preparing values
    #data = pd.read_excel( os.path.join(path_gcb, 'Global_Carbon_Budget_2022v1.0.xlsx'), sheet_name='Historical Budget' )
    data = pd.read_excel( os.path.join(path_gcb, 'Global_Carbon_Budget_2023v1.0.xlsx'), sheet_name='Historical Budget' )
    dd = np.array( data.values )
    ind_min = np.where( dd == 'Year' )[0][0]

    # creating dataset
    ds = xr.Dataset()
    ds.coords['year'] = dd[ind_min+1:,0]
    ds['FF_CO2'] = xr.DataArray( np.array(dd[ind_min+1:,1], dtype=np.float64), dims=('year',) )
    ds['LUC_CO2'] = xr.DataArray( np.array(dd[ind_min+1:,2], dtype=np.float64), dims=('year',) )
    ds['cement_CO2'] = xr.DataArray( np.array(dd[ind_min+1:,6], dtype=np.float64), dims=('year',) )
    
    # adapting for use
    ds['FF_CO2'] *= 44/12 * 1.e3
    ds['LUC_CO2'] *= 44/12 * 1.e3
    ds['FF_CO2'].attrs['units'] = 'MtCO2/yr' # GtC/yr --> MtCO2/yr
    ds['LUC_CO2'].attrs['units'] = 'MtCO2/yr' # GtC/yr --> MtCO2/yr
    ds['FF_CO2'].attrs['info'] = 'fossil emissions excluding carbonation'
    return ds
#---------------------------------------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------------------------------------
# Regions World Bank adapted
def func_prepare_regions( path_reg ):
    # adding directly a dictionary for specific issues
    dico_exceptions_countries = {"Cote d'Ivoire":["Côte d'Ivoire"]}
    
    # creating outputs
    dico_ISO2country = {}
    dico_country2reg = {}
    with open( os.path.join(path_reg, 'world-regions-according-to-the-world-bank.csv'), newline='') as csvfile:
        read = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(read):
            if i>0:
                if row[1] not in dico_ISO2country: # several lines with same ISO for countries with different names. First one is the most appropriate one.
                    dico_ISO2country[ row[1] ] = row[0]
                dico_country2reg[ row[0] ] = row[4]
                if row[0] in dico_exceptions_countries:
                    for key in dico_exceptions_countries[row[0]]:
                        dico_country2reg[key] = row[4]
    return dico_ISO2country, dico_country2reg
#---------------------------------------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------------------------------------
# REGRIDDING
def regrid_cdo( path_file_in, path_file_out, path_grid, method):
    if method == 'bil': # 'Bilinear interpolation'
        CDO.remapbil( path_grid, options='-b F64', input=path_file_in, output=path_file_out )
        
    elif method == 'con2': # 'Second order conservative remapping'
        CDO.remapcon2( path_grid, options='-b F64', input=path_file_in, output=path_file_out )
        
    elif method == 'con': # # 'First order conservative remapping'
        CDO.remapcon( path_grid, options='-b F64', input=path_file_in, output=path_file_out )
        
    elif method == 'dis': # 'Distance-weighted average remapping'
        CDO.remapdis( path_grid, options='-b F64', input=path_file_in, output=path_file_out )
    else:
        raise Exception("Not prepared")
#---------------------------------------------------------------------------------------------------------------------------