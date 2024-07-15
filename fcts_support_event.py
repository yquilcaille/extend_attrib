import os
import unicodedata
from difflib import SequenceMatcher

import cartopy.crs as ccrs
import cftime as cft
import geopandas as gpd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regionmask
import seaborn as sns  # # for colors
import xarray as xr
from shapely.geometry import mapping
from shapely.ops import unary_union
from statsmodels.stats.weightstats import DescrStatsW

CB_color_cycle = sns.color_palette( 'colorblind', n_colors=10000 )
from fcts_support_basic import *

#import warnings
#warnings.simplefilter("ignore") # mostly due to HDF stuff from dealing with CMIP6




#---------------------------------------------------------------------------------------------------------------------------
# SPATIAL INFORMATION
# Acknowledging: https://github.com/regionmask/regionmask/issues/38
def sample_coord(coord):
    # sample coords for the percentage overlap

    d_coord = coord[1] - coord[0]

    n_cells = len(coord)

    left = coord[0] - d_coord / 2 + d_coord / 20
    right = coord[-1] + d_coord / 2 - d_coord / 20

    return np.linspace(left, right, n_cells * 10)

def mask_3D_frac_approx(regions, lon, lat, **kwargs):
    """Sample with 10 times higher resolution.

    """

    backend = regionmask.core.mask._determine_method(lon, lat)
    if "rasterize" not in backend:
        raise ValueError("'lon' and 'lat' must be 1D and equally spaced.")

    if np.min(lat) < -90 or np.max(lat) > 90:
        raise ValueError("lat must be between -90 and +90")

    lon_name = getattr(lon, "name", "lon")
    lat_name = getattr(lat, "name", "lat")

    lon_sampled = sample_coord(lon)
    lat_sampled = sample_coord(lat)

    mask_resampled = regions.mask_3D(lon_sampled, lat_sampled, **kwargs)

    # this is almost as fast as via a pure numpy function
    mask = mask_resampled.coarsen(lon=10, lat=10).mean()

    mask_reshaped = mask_resampled.values.reshape(-1, lat.size, 10, lon.size, 10)
    # that's the pure numpy function
    # mask = mask_reshaped.mean(axis=(2, 4))

    # maybe fix edges
    sel = np.abs(lat_sampled) <= 90
    if kwargs.get("wrap_lon") is not False and sel.any():

        e1 = mask_reshaped[:, 0].mean(axis=(1, 3), where=sel[:10].reshape(-1, 1, 1))
        e2 = mask_reshaped[:, -1].mean(axis=(1, 3), where=sel[-10:].reshape(-1, 1, 1))

        mask.data[:, 0] = e1
        mask.data[:, -1] = e2

    coords = {lon_name: lon, lat_name: lat}
    return mask.assign_coords(coords)
#---------------------------------------------------------------------------------------------------------------------------






#---------------------------------------------------------------------------------------------------------------------------
# TREATMENT OF THE EVENT
class treat_event:
    """
    Class used to identify and prepare timeseries for EM-DAT database

    Parameters
    ----------
    evt: xarray.Dataset
        Element of the EM-DAT database
        
    geobounds: geopandas.geodataframe.GeoDataFrame
        geographical boundaries, output from 'func_prepare_geobounds'
    
    dict_geobounds: dict
        geographical boundaries, output from 'func_prepare_geobounds'
    
    threshold_SequenceMatcher: float between 0 and 1
        If defining the event using the 'reported' option, need to match the names of geobounds to those reported in the EM-DAT database.
        However, some names are not written similarly in geobounds and in EM-DAT: e.g. Stata Zagora and Stara Zagora. The solution is to use SequenceMatcher with this parameter.
        
    delta_days: int
        Analyzing the strength of events occuring over a very short period (e.g. 1 day) requires to look at the extent of potential events at this time of the year. 
        Explicitely, with an event defined over {(y0, m0, d0), (y1, m1, d1)}, instead of sampling every year on {(yr, m0, d0), (yr + y1 - y0, m1, d1)}, produce a rolling mean {(yr, m0, d0 - delta_days), (yr + y1 - y0, m1, d1 + delta_days)}.
        Depending on the type of event, will either take the minimum (e.g. cold wave) or the maximum (e.g. heat wave).
    """
    
    # --------------------
    # INITIALIZATION
    def __init__( self, evt, geobounds, dict_geobounds, threshold_SequenceMatcher=0.85, add_roll_windows=[], option_detailed_prints=False ):
        # preparing key variables
        self.evt = evt
        self.identifier_event = str(self.evt['DisNo.'].values)
        self.geobounds = geobounds
        self.dict_geobounds = dict_geobounds
        self.warnings = {}
        self.option_detailed_prints = option_detailed_prints
        
        # preparing other attributes
        self.threshold_SequenceMatcher = threshold_SequenceMatcher
        tmp = list( add_roll_windows )
        tmp.sort(reverse=True)
        self.windows = ['not filled yet'] + tmp #
        
        # colors
        self.colors_windows = [CB_color_cycle[i] for i, window in enumerate(self.windows)]

        # identifying type of event
        self.def_event_type()
        
        # identifying period of interest
        self.def_period()
    # --------------------

    


    # --------------------
    # IDENTIFICATION
    def def_event_type( self ):
        # identifying event
        self.event_type = [ self.clean_spaces(str(self.evt['Disaster Type'].values)), self.clean_spaces(str(self.evt['Disaster Subtype'].values)) ]
        
        # message for info
        if self.option_detailed_prints:
            print( "Type of event: "+self.event_type[0]+" ("+self.event_type[1]+")" )
        
        # identifying if low or high type
        if self.event_type[0] == 'Extreme temperature':
            self.event_type.append( {'Cold wave':'min', 'Severe winter conditions':'min', 'Heat wave':'max'}[self.event_type[1]] )
        else:
            raise Exception("Unprepared type of disaster")
    
    
    def def_period( self ):
        """
            Reading reported dates for the event. If values are missing, may have to fill them in under some conditions.
        """
        # initialize values that will be output
        time_values = {'Start':{'Year':None, 'Month':None, 'Day':None},
                       'End':{'Year':None, 'Month':None, 'Day':None}}
        
        # behavior if missing values
        behavior_if_not_there = {'Year':'Exception',
                                 'Month':{'Start':1, 'End':12},
                                 'Day':{'Start':1, 'End':self.n_days} }
        
        # loop over Year, Month, Day
        for dt in ['Year', 'Month', 'Day']:
            # checking if only one time value has been provided
            #if (np.isnan( float(self.evt['Start '+dt].values) ) and np.isnan( float(self.evt['End '+dt].values) )==False)  or \
            #(np.isnan( float(self.evt['Start '+dt].values) )==False and np.isnan( float(self.evt['End '+dt].values) )):
            #    raise Exception("Reporting error: only one of two "+dt+"s has been provided: what do do there..? fill using behavior_if_not_there?")
            # event 6492: example where only End Day is not reported. Decides to use last day there.
            
            # checking if no time value has been provided
            for tm in time_values.keys():
                if np.isnan( float(self.evt[tm+' '+dt].values) ):
                    if behavior_if_not_there[dt] == 'Exception':
                        raise Exception("Necessary to have values provided for "+dt)

                    elif type(behavior_if_not_there[dt]) == dict:
                        if type( behavior_if_not_there[dt][tm] ) == int:
                            time_values[tm][dt] = behavior_if_not_there[dt][tm]
                            
                        else:
                            time_values[tm][dt] = behavior_if_not_there[dt][tm](yr=time_values[tm]['Year'], m=time_values[tm]['Month'])
                        if self.option_detailed_prints:
                            print("WARNING: had to fill in "+tm+" "+dt)
                        if 'time_'+tm not in self.warnings:
                            self.warnings['time_'+tm] = []
                        self.warnings['time_'+tm].append( dt )
                 
                # values are both there, yay!
                else:
                    time_values[tm][dt] = int( self.evt[tm+' '+dt].values )
        
        # event 11076 has a heatwave reported over more than a year. correcting such things:
        if (self.format_date( time_values['End'] ) - self.format_date( time_values['Start'] )) > 365:
            time_values['End']['Year'] = time_values['Start']['Year']
        
        # finish period
        self.event_period = time_values
        self.event_year = self.event_period['Start']['Year']
        if self.option_detailed_prints:
            print('Event defined over: ' + str(self.format_date(dict_date=self.event_period['Start'])) + ' to ' + str(self.format_date(dict_date=self.event_period['End'])) )
        
        # deducing the characteristic length of the event
        length_event = (self.format_date( self.event_period['End'] ) - self.format_date( self.event_period['Start'] ) ) / np.timedelta64(1, 'D')
        if ('time_Start' not in self.warnings) and ('time_End' not in self.warnings):
            # will do a mean over the period as such
            self.windows[0] = 'mean'
            
        elif ('time_Start' in self.warnings) and ('time_End' in self.warnings) or\
        ('time_Start' in self.warnings) and ('time_End' not in self.warnings) or\
        ('time_Start' not in self.warnings) and ('time_End' in self.warnings): # could be written just "else", but keeping that open to add number of missing days
            # will do a running mean over the period, and select the extremum within
            # The average of reported heatwaves below 1 month is 8.48 days
            # if declared over same month, take 8 days
            # if declared over 2 months, take 45 days
            # etc.
            n_days = (int(np.round(length_event/30,0))-1) * 30 + 8 # np.round because of february
            self.windows[0] = self.event_type[2] + str(n_days) + 'days'        
            
            
    def def_spatial_units( self ):
        """
             Read the reported location of the event in the EM-DAT database. This text is matched to the names of the spatial units in the geobounds database associated with the reported country.
             To do so, several processes are employed when comparing names in the reported location to names of spatial units in the geobounds database
                 - Accents: some names may have accents, while their counterpart may not (e.g. Zürich vs Zurich) --> accents are removed from names
                 - Characters: some characters may or may not be present (e.g. Pays de la Loire vs Pays-de-la-Loire) --> replacing them with spaces
                 - Spellings: some names are not written similarly (e.g. Stata Zagora and Stara Zagora) --> using SequenceMatcher with 'threshold_SequenceMatcher'
                     (good e.g.'Split-Dalmatija' & 'Split-Dalmacia' ~ 0.897; 'Jharkland' & 'Jharkhand' ~ 0.889)
                     (Some names are similar but still too far apart, not lowering more the threshold to avoid false positives:
                     (bad e.g. 'Kavadartsi' & 'Kavadarci' ~ 0.842, 'Strumitsa' & 'Strumica' ~ 0.824, 'Vinitsa' & 'Vinica' ~ 0.769, 'Vâlcea' & 'Vilcea' ~ 0.833)
                     (Not using "word in location", because the SequenceMatcher helps with these cases, and create false True: e.g. Parana in location, but also Para region not wanted.)
                 - Lack of spaces: may have spaces forgotten (e.g. "Noord-holland,Overijssel") --> introduce spaces when missing
                 - Regional levels: different levels of regional aggregation are reported (e.g. canton of Aargau & city of Luzern) --> using 1st, 2nd level of geobounds
                 - Variants of names: some names have variants (e.g. Varaždinska, Varaždin, Varasd) --> using as well the variants reported in geobounds
                 - Region: some countries (e.g. Makedonia) put the name of the region in REGION --> using this one as well
                 - Very detailed levels: some events are defined at smaller scale, going up to 4th level (e.g. Cleveland, GBR) --> going up to this level only when discover required, to limit risks of false positives.
                 - Bad variants of names: some variants cannot be used as such --> keeping only those with more than 2 letters
                     (e.g. Comunidad Valenciana with variants Valencia|Communauté de Valence|C)
                     (e.g. Moravskoslezský with variants Mährisch-Ostrau|Mähren|Morava|Mo)
                     (e.g. Comunidad Foral de Navarra with variants Communauté forale de Navarre|Com)
                     (BUT Romania has regions with 3 letters, e.g. Boj & Olt --> setting Com of Spain as exception, and limit at 2 letters)
                 - Lower/Uppercases: discrepancies in lower/uppercases (e.g. Bjelovar-bilogora vs Bjelovarska-Bilogorska) --> all to lowercases
                 - Useless keywords: (e.g. 'Santa Cruz' in geobounds, but 'Santaz Cruz provinces' in Location) --> removing such keywords, singular & plural
                 - Phrasing: different ways to write spatial units --> two way reading
                     1. some words may be added in reported location, (e.g. provinces), but with too much of an impact to use SequenceMatcher
                     --> check which spatial unit of geobounds are within the reported location (e.g. 'Santa Cruz' in geobounds, but 'Santaz Cruz provinces' in Location)
                     2. some words may be added in geobounds (e.g. suburbs of...), also with too much of an impact to use SequenceMatcher
                     --> check which element of location are within names of spatial units in geobounds (e.g. 'Buenos Aires' in Location, but 'Ciudad de Buenos Aires' in geobounds)
                     (with previous improvements, 2. may not be useful anymore, and besides, longer step. To check. Could still be worthwhile to ensure better quality of recognition.)
                     (actually creating issues: Koch Bihar is not in the state of Bihar...)
                     (Auvergne in location, but Auvergne-Rhônes-Alpes: solved by 2nd, not by 1st)
                 - Exceptions: additional stuff such as changes in names of regions (e.g. 'Champagne-Ardenne' in location vs 'Grand Est' or 'Ardennes' in geobounds) --> dictionary to keep track of issues
                     (dico_correct_EMDAT: replacing these regions. Also used for correcting names too far apart.)
                 - Disputed regions: adding them if necessary
                 - Similar names: may have a region sharing name of another spatial unit in another region: exceptions
                     (e.g. Parana province vs Paraúna, Paraná, Paranã districts that are in regions goias, rio grande do norte, tocantins)
                 - Undeclared ISOs: Canaries islands is not SPI but Spain, ESP.
            
        """
        # Defining country and cleaning its text for next step
        self.country = str(self.evt['Country'].values)
        clean_country = self.clean_text( self.country, remove_useless_keywords=False, correct_EMDAT=False )

        # defining ISO, with some corrections if issues (undeclared ISO, disputed territories) --> define self.event_iso
        self.verify_iso( clean_country )
        
        # checking reported sub-regions in Location, and cleaning the string of location
        self.location_init = str(self.evt['Location'].values)
        location = self.clean_text( self.location_init )
        
        # considering regions in the country AND disputed regions
        list_all_subisos = np.hstack( [self.dict_geobounds[iso] for iso in self.event_iso] )

        if location == 'nan':
            if self.option_detailed_prints:
                print('WARNING: no location reported here, taking full country here: '+self.country)
            self.warnings['location'] = 'no location reported here, taking full country here: '+self.country

            # preparing all elements
            self.event_subunits = list_all_subisos
            self.event_subnames = list( set(self.geobounds.loc[self.event_subunits]['NAME_1']) )
            self.event_subnames.sort()

            # creating final region
            self.event_region = gpd.GeoSeries( unary_union(self.geobounds.loc[self.event_subunits]['geometry']) )
            
        elif location in ['all country', 'entire country']:
            # preparing all elements
            self.event_subunits = list_all_subisos
            self.event_subnames = list( set(self.geobounds.loc[self.event_subunits]['NAME_1']) )
            self.event_subnames.sort()

            # creating final region
            self.event_region = gpd.GeoSeries( unary_union(self.geobounds.loc[self.event_subunits]['geometry']) )

        else:

            # checking which subisos of geobounds are in the string of Location
            # some may not be found: e.g. 'Buenos Aires' in Location, but 'Ciudad de Buenos Aires' in geobounds would not be taken --> need complementary check
            self.event_subunits = []
            for i_subiso in list_all_subisos:

                # list of words in geobounds to try:
                list_words = [self.geobounds.loc[i_subiso]['NAME_1']] + self.geobounds.loc[i_subiso]['VARNAME_1'].split('|') + [self.geobounds.loc[i_subiso]['NAME_2']] + self.geobounds.loc[i_subiso]['VARNAME_2'].split('|') + [self.geobounds.loc[i_subiso]['REGION']]
                # specific cases, limited to some countries to avoid false positives
                if self.event_iso[0] in ['GBR']:
                    list_words = list_words + [self.geobounds.loc[i_subiso]['NAME_3']] + self.geobounds.loc[i_subiso]['VARNAME_3'].split('|') + [self.geobounds.loc[i_subiso]['NAME_4']] + self.geobounds.loc[i_subiso]['VARNAME_4'].split('|')
                elif self.event_iso[0] in ['PAK']:
                    list_words = list_words + [self.geobounds.loc[i_subiso]['NAME_3']] + self.geobounds.loc[i_subiso]['VARNAME_3'].split('|')

                # removing some exceptions really tricky
                list_words = self.remove_exceptions( list_words )

                # checking whether there is a match
                test_add_iso, counter = False, 0
                while (test_add_iso==False) and (counter<len(list_words)):
                    word = self.clean_text( list_words[counter], correct_EMDAT=False )
                    if (len(word) > 2) and np.any([SequenceMatcher(None, word, t).ratio() > self.threshold_SequenceMatcher for t in location.split(', ')]): # not keeping "word in location" due to Para/Parana
                        test_add_iso = True
                    else:
                        counter += 1

                # adds only if worked
                if test_add_iso:
                    self.event_subunits.append( i_subiso )

            ## taking ensemble of two sets#self.event_subunits = np.unique( self.list_subisos_geo2loc + self.list_subisos_loc2geo )
            self.event_subnames = list( set(self.geobounds.loc[self.event_subunits]['NAME_1']) )
            self.event_subnames.sort()
            if len(self.event_subunits) == 0:
                raise Exception('Nothing found here!')
            else:
                if self.option_detailed_prints:
                    print('Location of event reported in: '+self.location_init)
                    print('Location corrected for use as: '+location)
                    print('Identified geographical units: '+', '.join(self.event_subnames))
                self.test_entire_country = len(self.event_subunits) == len(list_all_subisos)
                if self.test_entire_country and self.option_detailed_prints:
                    print('(Entire country: '+self.country+'!)')
                # creating final region
                self.event_region = gpd.GeoSeries( unary_union(self.geobounds.loc[self.event_subunits]['geometry']) )
    # --------------------
    
    
    # --------------------
    # SUPPORT FOR TREATMENT
    def remove_exceptions( self, list_w ):
        # BRAZIL
        if self.event_iso[0] == 'BRA':
            if ('Parana province' in self.location_init):
                for w in ['Paraúna', 'Paranã', 'Paraná']:
                    # removing only if not the first one: region vs district
                    if (w in list_w) and (list_w.index(w) != 0):
                        list_w.remove( w )
                raise Exception('Here, should rewrite that with remove_districts_pb when it pops up')
                    
        # CHINA
        elif self.event_iso[0] == 'CHN':
            for_remove_districts_pb = [
                {'spatial_req':'Guangzhou',    'spatial_not_req':'Ganzhou',    'spatial_unless':['Jiangxi', 'Jiāngxī']},\
                {'spatial_req':'Guangzhou',    'spatial_not_req':'Gànzhōu',    'spatial_unless':['Jiangxi', 'Jiāngxī']},\
            ]
            for case in for_remove_districts_pb:
                if case['spatial_req'] in self.location_init:
                    list_w = self.remove_districts_pb( list_w=list_w, spatial_req=case['spatial_req'], spatial_not_req=case['spatial_not_req'], spatial_unless=case['spatial_unless'] )
                
                
        # INDIA
        elif self.event_iso[0] == 'IND':
            for_remove_districts_pb = [
                {'spatial_req':'Madhya Pradesh',    'spatial_not_req':'Andhra Pradesh',    'spatial_unless':['Andhra Pradesh']},\
                {'spatial_req':'Andhra Pradesh',    'spatial_not_req':'Madhya Pradesh',    'spatial_unless':['Madhya Pradesh']},\
                {'spatial_req':'West Rajasthan',    'spatial_not_req':'Ālǐ',    'spatial_unless':['Xizang']},\
            ]
            for case in for_remove_districts_pb:
                if case['spatial_req'] in self.location_init:
                    list_w = self.remove_districts_pb( list_w=list_w, spatial_req=case['spatial_req'], spatial_not_req=case['spatial_not_req'], spatial_unless=case['spatial_unless'] )

                    
        # JAPAN
        elif self.event_iso[0] == 'JPN':
            for_remove_districts_pb = [
                {'spatial_req':'Aichi',    'spatial_not_req':'Achi',    'spatial_unless':['Nagano']},\
                {'spatial_req':'Kagawa',    'spatial_not_req':'Kamogawa',    'spatial_unless':['Chiba']},\
                {'spatial_req':'Kagawa',    'spatial_not_req':'Sukagawa',    'spatial_unless':['Fukushima']},\
                {'spatial_req':'Kagawa',    'spatial_not_req':'Akaigawa',    'spatial_unless':['Hokkaido']},\
                {'spatial_req':'Kagawa',    'spatial_not_req':'Fukagawa',    'spatial_unless':['Hokkaido']},\
                {'spatial_req':'Kagawa',    'spatial_not_req':'Nakagawa',    'spatial_unless':['Hokkaido', 'Nagano', 'Tochigi']},\
                {'spatial_req':'Kagawa',    'spatial_not_req':'Kanagawa',    'spatial_unless':['Kanagawa']},\
                {'spatial_req':'Kagawa',    'spatial_not_req':'Kakegawa',    'spatial_unless':['Shizuoka']},\
                {'spatial_req':'Kagawa',    'spatial_not_req':'Kozagawa',    'spatial_unless':['Wakayama']},\
                {'spatial_req':'Kumagaya',    'spatial_not_req':'Kamagaya',    'spatial_unless':['Chiba']},\
                {'spatial_req':'Saga',    'spatial_not_req':'Aga',    'spatial_unless':['Niigata']},\
                {'spatial_req':'Saga',    'spatial_not_req':'Sagae',    'spatial_unless':['Yamagata']},\
                {'spatial_req':'Tokyo',    'spatial_not_req':'Tōyō',    'spatial_unless':['Kochi']}
            ]
            for case in for_remove_districts_pb:
                if case['spatial_req'] in self.location_init:
                    list_w = self.remove_districts_pb( list_w=list_w, spatial_req=case['spatial_req'], spatial_not_req=case['spatial_not_req'], spatial_unless=case['spatial_unless'] )
        
        # MEXICO
        elif self.event_iso[0] == 'MEX':
            if ('Hidalgo' in self.location_init): # province Hidalgo BUT several districts Hidalgo in Coahuila / Michoacán / Tamaulipas
                list_w = self.remove_districts_pb( list_w=list_w, spatial_req='Hidalgo', spatial_not_req='Hidalgo', spatial_unless=['Coahuila', 'Michoacán', 'Tamaulipas'] )

        # PERU
        elif self.event_iso[0] == 'PER':
            if ('Ucayali' in self.location_init):
                # removing the district Ucayali in Loreto, if not required by Loreto
                list_w = self.remove_districts_pb( list_w=list_w, spatial_req='Ucayali', spatial_not_req='Ucayali', spatial_unless=['Loreto'] )
                
        # ROUMANIA
        elif self.event_iso[0] == 'ROU':
            for_remove_districts_pb = [
                {'spatial_req':'Giurgiu',    'spatial_not_req':'Gurghiu',    'spatial_unless':['Mureș']},\
                {'spatial_req':'Galati',    'spatial_not_req':'Galateni',    'spatial_unless':['Teleorman']},\
            ]
            for case in for_remove_districts_pb:
                if case['spatial_req'] in self.location_init:
                    list_w = self.remove_districts_pb( list_w=list_w, spatial_req=case['spatial_req'], spatial_not_req=case['spatial_not_req'], spatial_unless=case['spatial_unless'] )
                        
        # RUSSIA
        elif self.event_iso[0] == 'RUS':
            if ('Moskovskaya' in self.location_init):
                list_w = self.remove_districts_pb( list_w=list_w, spatial_req='Moskovskaya', spatial_not_req='Pskovskaya Oblast', spatial_unless=['Pskov'] )
                        
        # SPAIN
        elif self.event_iso[0] == 'ESP':
            if ('Com' in list_w):
                list_w.remove( 'Com' )
                
        # UNITED KINGDOM
        elif self.event_iso[0] == 'GBR':
            if ('Manchester' in self.location_init):
                list_w = self.remove_districts_pb( list_w=list_w, spatial_req='Manchester', spatial_not_req='Lanchester', spatial_unless=['England'] )
                
        # USA
        elif self.event_iso[0] == 'USA':
            # Many cases where a state or district (e.g. Delaware) is required, BUT several counties share an identical name Delaware in other states (e.g. Indiana, Iowa, Ohio, Oklahoma)
            # funniest one: state of Kansas is required, but not the entirety of the Arkansas... ^^'
            for_remove_districts_pb = [
                {'spatial_req':'Northern California',    'spatial_not_req':'Butte',    'spatial_unless':['South Dakota']} ,\
                {'spatial_req':'Western Nevada',    'spatial_not_req':'Carson',    'spatial_unless':['Texas']} ,\
                {'spatial_req':'Delaware',   'spatial_not_req':'Delaware',   'spatial_unless':['Indiana', 'Iowa', 'Ohio', 'Oklahoma']} ,\
                {'spatial_req':'Western Nevada',    'spatial_not_req':'Douglas',    'spatial_unless':['Colorado', 'Georgia', 'Illinois', 'Kansas', 'Minnesota', 'Missouri', 'Nebraska', 'South Dakota', 'Wisconsin']} ,\
                {'spatial_req':'Northern California',    'spatial_not_req':'Humboldt',    'spatial_unless':['Iowa']} ,\
                {'spatial_req':'Indiana',    'spatial_not_req':'Indiana',    'spatial_unless':['Pennsylvania']} ,\
                {'spatial_req':'Iowa',       'spatial_not_req':'Kiowa',      'spatial_unless':['Colorado', 'Kansas', 'Oklahoma']} ,\
                {'spatial_req':'Iowa',       'spatial_not_req':'Iowa',      'spatial_unless':['Wisconsin']} ,\
                {'spatial_req':'Kansas',   'spatial_not_req':'Arkansas',   'spatial_unless':['Arkansas']} ,\
                {'spatial_req':'Mississippi',   'spatial_not_req':'Mississippi',   'spatial_unless':['Arkansas']} ,\
                {'spatial_req':'Nevada',   'spatial_not_req':'Nevada',   'spatial_unless':['Arkansas', 'California']} ,\
                {'spatial_req':'Northern California',    'spatial_not_req':'King',    'spatial_unless':['Texas']} ,\
                {'spatial_req':'Northern California',    'spatial_not_req':'Kings',    'spatial_unless':['New York']} ,\
                {'spatial_req':'Northern California',    'spatial_not_req':'Lake',    'spatial_unless':['Colorado', 'Florida', 'Illinois', 'Indiana', 'Michigan', 'Minnesota', 'Montana', 'Ohio', 'South Dakota', 'Tennessee']} ,\
                {'spatial_req':'Northern California',    'spatial_not_req':'Leake',    'spatial_unless':['Mississippi']} ,\
                {'spatial_req':'Northern California',    'spatial_not_req':'Marion',    'spatial_unless':['Alabama', 'Arkansas', 'Florida', 'Georgia', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Mississippi', 'Missouri',\
                                                                                                         'Ohio', 'South Carolina', 'Tennessee', 'Texas', 'West Virginia']} ,\
                {'spatial_req':'Northern California',    'spatial_not_req':'Martin',    'spatial_unless':['Florida', 'Indiana', 'Kentucky', 'Minnesota', 'North Carolina', 'Texas']} ,\
                {'spatial_req':'Northern California',    'spatial_not_req':'Nevada',    'spatial_unless':['Arkansas']} ,\
                {'spatial_req':'Northern California',    'spatial_not_req':'Santa Cruz',    'spatial_unless':['Arizona']} ,\
                {'spatial_req':'Northern California',    'spatial_not_req':'Sierra',    'spatial_unless':['New Mexico']} ,\
                {'spatial_req':'Northern California',    'spatial_not_req':'Trinity',    'spatial_unless':['Texas']} ,\
                {'spatial_req':'North Carolina',    'spatial_not_req':'South Carolina',    'spatial_unless':['South Carolina']} ,\
                {'spatial_req':'Ohio',       'spatial_not_req':'Ohio',       'spatial_unless':['Kentucky', 'Indiana', 'West Virginia']} ,\
                {'spatial_req':'Oregon',    'spatial_not_req':'Oregon',    'spatial_unless':['Missouri']} ,\
                {'spatial_req':'Texas',      'spatial_not_req':'Texas',      'spatial_unless':['Missouri', 'Oklahoma']} ,\
                {'spatial_req':'Washington',      'spatial_not_req':'Washington',      'spatial_unless':['Alabama', 'Arkansas', 'Colorado', 'Florida', 'Georgia', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky',\
                                                                                                        'Louisiana', 'Maine', 'Maryland', 'Minnesota', 'Mississippi', 'Missouri', 'Nebraska', 'New York', 'North Carolina', 'Ohio',\
                                                                                                        'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Wisconsin']},\
                {'spatial_req':'Western Nevada',    'spatial_not_req':'Lyon',    'spatial_unless':['Iowa', 'Kansas', 'Kentucky', 'Minnesota']} ,\
                {'spatial_req':'Western Nevada',    'spatial_not_req':'Story',    'spatial_unless':['Iowa']} ,\
            ]

            for case in for_remove_districts_pb:
                if case['spatial_req'] in self.location_init:
                    list_w = self.remove_districts_pb( list_w=list_w, spatial_req=case['spatial_req'], spatial_not_req=case['spatial_not_req'], spatial_unless=case['spatial_unless'] )
        return list_w
    
    def remove_districts_pb( self, list_w, spatial_req, spatial_not_req, spatial_unless ):
        # state spatial_req BUT several counties also called spatial_req in a list of spatial_unless
        if spatial_req in self.location_init:
            # removing the district spatial_req in spatial_unless, if not required by spatial_unless through location
            for district_pb in spatial_unless:
                #if (district_pb not in self.location_init):
                if (district_pb not in self.location_init) and (list_w[0] == district_pb):
                    while (spatial_not_req in list_w):# while and not if: may have more than 1, like Arkansas...
                        list_w.remove( spatial_not_req )
        return list_w
    
    def eval_ISO_panorama(self):
        # function only used for panorama, to be applied directly on EMDAT events and avoid gathering all results
        self.country = str(self.evt['Country'].values)
        clean_country = self.clean_text( self.country, remove_useless_keywords=False, correct_EMDAT=False )
        self.verify_iso( clean_country )
        return self.event_iso[0]
        

    def verify_iso(self, clean_country):
        # Basic ISO, enough most of the time
        out_iso = [str(self.evt['ISO'].values)]
        
        # Edit ISO if necessary
        if (str(self.evt['Location'].values) == 'Canarias province')  and (out_iso == ['SPI']):
            out_iso = ['ESP']
            out_country = 'spain'
        elif (out_iso == ['SCG']):
            out_iso = ['SRB', 'MNE'] # 'Serbia Montenegro' split in 'Serbia' & 'Montenegro'
        
        # Prepare disputed territories
        self.dico_disputed = {  }
        for iso in self.dict_geobounds.keys():
            list_subisos = self.dict_geobounds[iso]
            if self.geobounds.loc[ list_subisos[0] ]['DISPUTEDBY'] != '':
                sovereign = self.geobounds.loc[ list_subisos[0] ]['SOVEREIGN']
                disputed = self.geobounds.loc[ list_subisos[0] ]['DISPUTEDBY'].replace( ' and' , ' ' ).split( ', ' )
                for country in [sovereign] + disputed:
                    country = self.clean_text( country, correct_EMDAT=False )
                    if country not in self.dico_disputed.keys():
                        self.dico_disputed[country] = []
                    self.dico_disputed[country].append( iso )
                    
        # Add disputed territories if necessary
        if clean_country in self.dico_disputed:
            self.event_iso = out_iso + self.dico_disputed[clean_country]
        else:
            self.event_iso = out_iso
    
    def clean_text( self, init_text, remove_useless_keywords=True, correct_EMDAT=True ):
        # basic operation: remove tricky characters like accented ones
        corrected_text = self.clean_characters( init_text )
        
        # basic operation: removing useless soaces, adding missing ones after ','
        corrected_text = self.clean_spaces( corrected_text )
        
        # basic operation: everything as lowercase to ease identification
        corrected_text = corrected_text.lower()

        # basic operation: remove words messing up with identification, like province or district
        if remove_useless_keywords:
            corrected_text = self.remove_useless_keywords( corrected_text )
        
        if correct_EMDAT:
            # advanced operation: replacing some elements in location, if not correctly reported in EM-DAT
            corrected_text = self.correct_EMDAT( corrected_text )
        return corrected_text
    
    @staticmethod
    def clean_spaces( string ):
        # 1. removing useless spaces
        if len(string) > 0:
            string = string.strip()
                
        # 2. removing double spaces
        new_string = ''
        for i_x in range(len(string)):
            if string[i_x] != ' ':
                # simply add the character
                new_string = new_string + string[i_x]

            elif (string[i_x+1] == ' '):
                # double space: skipping this space, and adding the next one
                pass

            else:
                # simply add this space
                new_string = new_string + string[i_x]
            
        # prepare next step
        string = new_string
                
        # 3. putting spaces where some got forgotten: "X,Y" --> "X, Y"
        new_string = ''
        for i_x in range(len(string)):
            if string[i_x] != ',':
                # simply add the character
                new_string = new_string + string[i_x]
            elif (i_x < len(string)-1) and (string[i_x+1] == ' '):
                # character is ',', add it, next iteration will add correctly the space
                new_string = new_string + ',' # string[i_x] == ','
            elif (i_x < len(string)-1) and (string[i_x+1] != ' '):
                # missing space! add ', ' before new character!
                new_string = new_string + ', '

        return new_string
        
    @staticmethod
    def clean_characters( string ):
        # replace specific characters e.g. accents
        nfkd_form = unicodedata.normalize('NFKD', string)
        new_string = u"".join([c for c in nfkd_form if not unicodedata.combining(c)])
        
        # need to replace additional characters:
        # - some characters are not filtered out, because not handled by unicode.normalize
        # - numbers have to be replaced with nothing
        # - additional characters have to be replaced with spaces
        dico_charac_pb = {'ı':'i', \
                          '0':'', '1':'', '2':'', '3':'', '3':'', '4':'', '5':'', '6':'', '7':'', '8':'', '9':'',\
                         '-':' ', '/':' ', '!':' ', '?':' ', ' ;':',', ';':','}
        for charac in dico_charac_pb:
            new_string = new_string.replace( charac, dico_charac_pb[charac] )
        return new_string
        
    @staticmethod
    def remove_useless_keywords( string_v0 ):
        list_reg_words = ['province', 'region', 'canton', 'department', 'district', 'oblast', 'state', 'city', 'cities', 'isl.', 'prefecture', 'borough', 'county', 'counties']
        # general thing: removing final word if about province or any other regional term
        # edit: not only final word, due to cases like: 'Chongqing Shi province, Shijiazhuang district (Hebei Sheng province)'
        string_v1, pieces = '', string_v0.split( ', ' )
        for i, reg in enumerate(pieces):
            # checking if one regional keyword is at the end, singular or plural, if so, remove it
            for keyword in list_reg_words:
                if reg[-(1 + len(keyword) + 1):] == ' ' + keyword + 's':
                    reg = reg[:-(1 + len(keyword) + 1)]
                if reg[-(1 + len(keyword)):] == ' ' + keyword:
                    reg = reg[:-(1 + len(keyword))]
            # appending that
            string_v1 += reg
            # adding as well the ', ' if not the last part
            if i != len(pieces)-1:
                string_v1 += ', '

        # specific case: simplifying the following cases: works for all, except when a ', ' is in the middle of the 1st or 2nd part. Not flawless, too many exceptions...
        string_v2, pieces = '', string_v1.split( ', ' )
        for i, reg in enumerate(pieces):
            reg_split = reg.split( ' (' )
            # checking that in one of those cases: is there a (?
            if len(reg_split) > 1:
                test_left = np.any( [keyword in reg_split[0] for keyword in list_reg_words] )
                test_right = np.any( [keyword in reg_split[1] for keyword in list_reg_words] )
                # checking that in one of those cases: is there something like X city (Y province)?
                if test_right:# test_left and test_right: ### REMOVING TEST_LEFT!!
                    # simplify by taking only left part (still being able to recognize this part), while removing the keyword
                    for keyword in list_reg_words:
                        if keyword in reg_split[0]:
                            if reg_split[0][-(1 + len(keyword) + 1):] == ' ' + keyword + 's':
                                piece = reg_split[0][:-(1 + len(keyword) + 1)]
                            elif reg_split[0][-(1 + len(keyword)):] == ' ' + keyword:
                                piece = reg_split[0][:-(1 + len(keyword))]
                            else:
                                piece = reg_split[0]
                        elif keyword in reg_split[1]:
                            piece = reg_split[0]
                    # adding this one
                    string_v2 += piece.strip()
                else:
                    string_v2 += reg
            else:
                string_v2 += reg
            # adding as well the ', ' if not the last part
            if i != len(pieces)-1:
                string_v2 += ', '
            
        return string_v2
    
    def correct_EMDAT(self, string_mod):
        # dictionaries for correcting entries of EMDAT
        # if name : name --> simple replace
        # if name : name1, name2 --> simple replace
        # if name : [name,ISO] --> replace only if in this ISO ('centre')
        # if name1|name2... : name --> replace name1,... by name
        # if name1|name2... : [name,ISO] --> replace name1,... by name only if in this ISO
        dico_correct_EMDAT = {
            'durres city (durres, durres province)': ['durres', 'ALB'],\
            'new south wales (south)': ['new south wales', 'AUS'],\
            'tasmania (northern)': ['tasmania', 'AUS'],\
            'region bruxelles capitale brussels hoofdtedelijk gewes': ['bruxelles', 'BEL'],\
            'region de bruxelles capitale brussels hoofdstedelijk gewes': ['bruxelles', 'BEL'],\
            'region wallonne': ['wallonie', 'BEL'],\
            'vlaams gewest': ['vlaanderen', 'BEL'],\
            'vlaams geweest': ['vlaanderen', 'BEL'],\
            'sofia': ['sofiya, grad sofiya', 'BGR'],\
            'bileca districts (republika srpska)': ['bileca', 'BIH'],\
            'capljina cities (neretvljanski, federacija bosne i hercegovine province)': ['capljina', 'BIH'],\
            'cocapata village (ayopaya, cochabamba province)': ['ayopaya', 'BOL'],\
            'quime village (inquisivi, la paz province)': ['inquisivi', 'BOL'],\
            'tinquipaya areas (tomas frias, potosi province)': ['tomas frias', 'BOL'],\
            'llallagua areas (rafael bustillo, potosi province)': ['rafael bustillo', 'BOL'],\
            'colquechaca area (chayanta, potosi province)': ['chayanta', 'BOL'],\
            'tomave area (antonio quijarro, potosi province)': ['antonio quijarro', 'BOL'],\
            'cariocas borough (nova lima, minas gerais province)': ['cariocas', 'BRA'],\
            'metropolitana': ['santiago metropolitan', 'CHL'],\
            'guangzhou city (guangzhou, guangdong sheng province)': ['guangzhou', 'CHN'],\
            'chongqing shi': ['chongqing', 'CHN'],\
            'shanghai shi': ['shanghai', 'CHN'],\
            'famgusta': ['famagusta, iskele', 'CYP'],\
            'nicosia': ['guzelyurt, nicosia', 'CYP'],\
            'jihomoravsky': ['jihomoravsky, kraj vysocina, zlinsky', 'CZE'],\
            'praha': ['prague', 'CZE'],\
            'severocesky': ['liberecky, ustecky', 'CZE'],\
            'severomoravsky': ['moravskoslezsky, olomoucky', 'CZE'],\
            'stredocesky': ['stredocesky', 'CZE'],\
            'vychodocesky': ['kralovehradecky, pardubicky', 'CZE'],\
            'zapadocesky': ['karlovarsky, plzensky', 'CZE'],\
            'zavaska': ['zasavska', 'CZE'],\
            'dresden city (dresden, sachsen province)': ['dresden', 'DEU'],\
            'brombach lake (mittelfranken, bayern province)': ['brombach', 'DEU'],\
            'adrar (province district )': ['adrar', 'DZA'],\
            'qena': ['qina', 'EGY'],\
            'cataluna catalunya': ['catalunya', 'ESP'],\
            'pais vasco euskadi': ['pais vasco', 'ESP'],\
            'auvergne|rhone alpes': ['auvergne rhone alpes', 'FRA'], \
            'nord pas de calais|picardie': ['hauts de france', 'FRA'],\
            'champagne ardenne': ['champagne ardennes', 'FRA'],\
            'alsace|lorraine|champagne ardennes': ['grand est', 'FRA'],\
            'haute normandie|basse normandie': ['normandie', 'FRA'],\
            'centre': ['centre val de loire', 'FRA', 'check if already there'],\
            'bourgogne|franche compte': ['bourgogne franche comte', 'FRA'],\
            'bourgogne|franche comte': ['bourgogne franche comte', 'FRA'],\
            'poitou charentes|limousin|aquitaine': ['nouvelle aquitaine', 'FRA'],\
            'midi pyrenees|languedoc rousillon': ['occitanie', 'FRA'],\
            "toute la france metropolitaine (excepte l'ouest de la bretagne et la corse)": ["auvergne rhone alpes, bourgogne franche comte, ille et vilaine, centre val de loire, grand est, hauts de france, ile de france, normandie, nouvelle aquitaine, occitanie, pays de la loire, provence alpes cote d'azur", 'FRA'],\
            'tyne and wear': ['gateshead, newcastle upon tyne, sunderland, north tyneside, south tyneside', 'GBR'],\
            'anatoliki makedonia kai thraki': ['east macedonia and thrace, athos', 'GRC'],\
            'attiki': ['attica', 'GRC'],\
            'corfu isl. (kerkyras, ionian islands province)': ['corfu', 'GRC'],\
            'dytiki ellada': ['west greece', 'GRC'],\
            'dytiki makedonia': ['west macedonia', 'GRC'],\
            'ionioi nisoi': ['ionian islands', 'GRC'],\
            'ipeiros': ['epirus', 'GRC'],\
            'kentriki makedonia': ['central macedonia', 'GRC'],\
            'kriti': ['crete', 'GRC'],\
            'peloponnisos': ['peloponnese', 'GRC'],\
            'sterea ellada': ['central greece', 'GRC'],\
            'thessalia': ['thessaly', 'GRC'],\
            'voreio aigaio': ['north aegean', 'GRC'],\
            'notio aigaio': ['south aegean', 'GRC'],\
            'orissa': ['odisha', 'IND', 'check if already there'],\
            'odisha (orissa)': ['odisha', 'IND'],\
            'vidarbha (=part of maharashtra)': ['amravati, akola, bhandara, buldhana, chandrapur, gadchiroli, gondja, nagpur, wardha, washim, yavatmal', 'IND'],\
            'delhi':['nct of delhi', 'IND', 'check if already there'],\
            'aurangabad, nawada districts (bihar)':['aurangabad, nawada', 'IND'],\
            'west rajasthan': ['barmer, bikaner, jaisalmer, jalore, jodhpur, nagaur, pali, rajsamand, sirohi, udaipur', 'IND'],\
            'northern':['hazafron, golan', 'ISR'],\
            'kooti': ['kochi', 'JPN'],\
            'kyooto provinces (honshu isl.)': ['kyooto', 'JPN'],\
            'saga provinces (kyushu isl.)': ['saga', 'JPN'],\
            'tokusima provinces (shikoko isl.)': ['tokusima', 'JPN'],\
            'tookyoo': ['tokyo', 'JPN'],\
            'kadamjay district in batken': ['kadamjay', 'KGZ'],\
            'kavadarci': ['kavadartsi', 'MKD'],\
            'strumica': ['strumitsa', 'MKD'],\
            'vinica': ['vinitsa', 'MKD'],
            'brod': ['makedonska brod', 'MKD'],\
            'maiduguri': ['borno', 'NGA'],\
            'nelson marlborough':['nelson, marlborough', 'NZL'],\
            'nagarparkar regions (tharparkar, sindh province)': ['nagarparkar', 'PAK'],\
            'guadra': ['guarda', 'PRT'],\
            'boj': ['dolj', 'ROU'],\
            'vilcea': ['valcea', 'ROU'],\
            'bucuresti': ['bucharest', 'ROU'],\
            'presov': ['presovsky', 'SVK'],\
            'zavaska': ['zasavska', 'SVN'],\
            'modesto city (stanislaus, california province)': ['stanislaus', 'USA'],\
            'northern california': ['alameda, alpine, amador, butte, calaveras, colusa, contra costa, del norte, el dorado, fresno, glenn, humboldt, inyo, kings, lake, lassen, madera, marin, mariposa, mendocino, merced, modoc, mono, monterey, napa, nevada, placer, plumas, sacramento, san benito, san francisco, san joaquin, san mateo, santa clara, santa cruz, shasta, sierra, siskiyou, solano, sonoma, stanislaus, sutter, tehama, trinity, tulare, tuolumne, yolo, yuba', 'USA'],\
            'western nevada': ['carson, douglas, lyon, storey, washoe', 'USA'],\
        }
        
        for item_in in dico_correct_EMDAT:
            list_it = item_in.split( '|' )

            # first case: replacing 1 region to 1 or more. Due to spelling, change in names, etc.
            if len(list_it) == 1:
                # replace only if in correct country, evaluated through ISO code
                if (item_in in string_mod) and (self.event_iso[0] == dico_correct_EMDAT[item_in][1]):
                    # need to double check if not already there: e.g. we want to replace in France 'centre' --> 'centre val de loire'
                    # BUT some rows have already 'centre val de loire'; avoiding 'centre val de loire val de loire'
                    # BESIDES we want to avoid disturbing other cases: e.g. Spain 'cataluna catalunya' --> 'catalunya' & 'pais vasco euskadi' --> 'pais vasco'
                    if (len(dico_correct_EMDAT[item_in]) == 3)  and  (dico_correct_EMDAT[item_in][2] == 'check if already there') and (dico_correct_EMDAT[item_in][0] in string_mod):
                        # in the case of 'centre' --> 'centre val de loire' but already having 'centre val de loire' --> nothing to do.
                        pass
                    else:
                        string_mod = string_mod.replace( item_in, dico_correct_EMDAT[item_in][0] )

            # second case: replacing several regions to 1. Due to aggregation of regions.
            else:
                # replace only if in correct country, evaluated through ISO code
                if np.all( [it in string_mod for it in list_it] ) and (self.event_iso[0] == dico_correct_EMDAT[item_in][1]):
                    for it in list_it:
                        # checking where to replace, and where to put comma to improve readibility
                        ind = string_mod.index( it )
                        # comma rather at the end
                        if ind > len(string_mod)/2:
                            comma = string_mod[ind-2:ind] == ', '
                            string_mod = string_mod[:ind-comma*2] + string_mod[ind+len(it):]
                        # comma rather at the beginning
                        else:
                            comma = string_mod[ind+len(it):ind+len(it)+2] == ', '
                            string_mod = string_mod[:ind] + string_mod[ind+len(it)+comma*2:]
                    # removed those that had to be removed, now put the new one
                    string_mod = string_mod + ', ' + dico_correct_EMDAT[item_in][0]
        return string_mod
                      
                      
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
            if 'T' in str(val0):
                tt = 'T'+str(val0).split('T')[1]
                return np.datetime64( '-'.join( [dic_dt['Year'], dic_dt['Month'], dic_dt['Day']] )+tt )
            else:
                return np.datetime64( '-'.join( [dic_dt['Year'], dic_dt['Month'], dic_dt['Day']] ), 'D')
        else:
            format_t = type(val0)
            dt = -1 * int((format_t in [cft.Datetime360Day]) and (dict_date['Day'] == 31))
            return format_t( year=dict_date['Year'], month=dict_date['Month'], day=dict_date['Day'] + dt, hour=val0.hour, minute=val0.minute, second=val0.second, microsecond=val0.microsecond)

    @staticmethod
    def n_days( yr, m ):
        d = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31 }[m]
        if (m == 2) and (yr % 4 == 0):
            d += 1
        return d
    
    def select_ESMs_CMIP6( self, comp_cmip6, n_ESMs_CMIP6 ):
        average_corr = self.spatial_average( comp_cmip6['corrcoef'], self.event_region )
        if False:
            average_reldif_stddev = self.spatial_average( (comp_cmip6['stddev'] - comp_cmip6['stddev_ref']) / comp_cmip6['stddev_ref'], self.event_region )
            # choosing those with the better combined standard deviation & correlation
            dist = np.sqrt( average_reldif_stddev ** 2 + (1 - average_corr)**2 )
        else:
            # choosing those with the better average correlation
            dist = 1 - average_corr
        indexes = dist.argsort().values[:n_ESMs_CMIP6]
        self.kept_esms = list( dist.esm.values[indexes] )
    # --------------------
    
    
    # --------------------
    # PLOT
    @staticmethod
    def title_on_lines( string, n_limit ):
        elts_string = string.split( ', ')
        string_lines, counter = '', 0
        for elt in elts_string:
            if counter > n_limit:
                string_lines = string_lines + '\n'
                counter = 0
            string_lines = string_lines + elt
            counter += len(elt)
            if elt != elts_string[-1]:
                string_lines = string_lines + ', '
                counter += 2
        return string_lines
    
    def do_suptitle( self, fig, figsize, fontsize ):
        # first part of the suptitle
        tmp = {}
        for tm in ['Start', 'End']:
            if 'time_'+tm in self.warnings:
                tmp[tm] = '.'.join([str(self.event_period[tm][s]) for s in self.event_period[tm] if s not in self.warnings['time_'+tm]])
            else:
                tmp[tm] = '.'.join([str(self.event_period[tm][s]) for s in self.event_period[tm]])
        ttl0 = self.event_type[1] + ' over ' + tmp['Start'] + " - " + tmp['End'] + ' in ' + self.country
        
        # second part of the suptitle
        if self.location_init not in ['nan']:
            # writing reported location
            long_ttl = ttl0 + ':\n' + self.title_on_lines(self.location_init, 100 * (figsize[0]/15) * (fontsize/15))
            
        else:
            long_ttl = ttl0
            
        # suptitle
        _ = fig.suptitle( long_ttl, fontsize=fontsize )

    
    def plt_reg(self, dic_spec_window, fig, obs_selecttime, fontsize, set_ratio_lon_lat=None):
        # preparing
        dic_ax_window, dic_dt = {}, {}

        # preparing mask: whole country
        list_all_subisos = np.hstack( [self.dict_geobounds[iso] for iso in self.event_iso] )
        country = gpd.GeoSeries( unary_union(self.geobounds.loc[list_all_subisos]['geometry']) )
        # could center on self.event_region instead of country, although lon_borders would not be properly defined
        mask = self.mask_reg( country, obs_selecttime[self.windows[0]].lon, obs_selecttime[self.windows[0]].lat )
        mask = mask.where( mask == 0, 1 ) # adapting to have all grid cells for correct min/max
        tmp = mask.where( mask>0, drop=True )
        lon_plot, lat_plot = tmp.lon.values, tmp.lat.values
        lon_plot[np.where(lon_plot>180)] -= 360

        # preparing central longitude and borders of map: longitude done in dirty way, but couldnt find better approach
        # SUPER MESSY PART THERE
        lat_borders = np.array( [np.min(lat_plot)-2.5, np.max(lat_plot)+2.5] )
        test_lon180 = 180 - np.max( [np.abs(np.min(lon_plot)),np.max(lon_plot)] ) < np.max( [np.abs(np.min(lon_plot)),np.max(lon_plot)] )
        if test_lon180:
            tmp_lon = np.copy(lon_plot)
            tmp_lon[np.where(lon_plot<0)] += 360
            lon_borders = np.array( [np.min(tmp_lon)-2.5, np.max(tmp_lon)+2.5] )
            central_longitude = np.mean(lon_borders)
            lon_borders[np.where(lon_borders>180)] -= 360
        else:
            lon_borders = np.array( [np.min(lon_plot+360)-2.5, np.max(lon_plot+360)+2.5] )
            central_longitude = np.mean(lon_borders)
        if lon_borders[0] > lon_borders[1]:
            lon_borders[0] -= 360
        if central_longitude > 180:
            central_longitude -= 360
        if (central_longitude < 0) and (obs_selecttime[self.windows[0]].lon.min() > 0):
            central_longitude += 360 # case USA on CMIP6-ng grid

        for window in self.windows:

            # centering projection (in case of countries stretching over -180 East: USA, New Zealand, etc)
            dic_ax_window[window] = plt.subplot( dic_spec_window[window], projection=ccrs.Robinson(central_longitude=central_longitude) )

            # ploting whole country
            reg = regionmask.Regions([country.item()])
            reg.plot(projection=ccrs.Robinson(), add_label=False, line_kws={'color':'black','linestyle':'-','lw':2}, add_coastlines=True, coastline_kws={'color':(0.25,0.25,0.25),'linestyle':'-','lw':2} )

            # ploting selected region
            reg = regionmask.Regions([self.event_region.item()])
            reg.plot(projection=ccrs.Robinson(), line_kws={'color':'red', 'linestyle':'-', 'lw':2}, add_label=False, add_coastlines=False)

            # identifying correct variable
            #dt = obs_selecttime[window][self.var_in_data(obs_selecttime[window])].sel( time=self.event_period['Start']['Year'], lat=y_lat ).isel( lon=ind_lon ).drop('time')
            #dt = obs_selecttime[window][self.var_in_data(obs_selecttime[window])].sel( time=self.event_period['Start']['Year'], lon=lon_plot, lat=lat_plot ).drop('time')
            dt = obs_selecttime[window][self.var_in_data(obs_selecttime[window])].sel( time=self.event_period['Start']['Year'] ).drop('time')
            if 'member' in dt.coords:
                dt = dt.mean('member')
            dic_dt[window] = dt.compute()

        # creating common colorbar scale
        vmin = np.min( [np.nanmin(dic_dt[window]*mask) for window in self.windows] )
        vmax = np.max( [np.nanmax(dic_dt[window]*mask) for window in self.windows] )

        # ploting
        for window in self.windows:
            pmesh = dic_ax_window[window].pcolormesh(obs_selecttime[window]['lon'], \
                                                     obs_selecttime[window]['lat'], \
                                                     dic_dt[window], \
                                                     transform=ccrs.PlateCarree(), rasterized=True, vmin=vmin, vmax=vmax )
            dic_ax_window[window].set_extent( (lon_borders[0], lon_borders[1], lat_borders[0], lat_borders[1]), crs=ccrs.PlateCarree())

            # adding title
            ttl0 = '$T_{'+window+'}$ of '+self.name_data
            if 'location' in self.warnings:
                ttl0 += '\nlocation not reported'
            dic_ax_window[window].text(-0.300, 0.55, s=ttl0, fontdict={'size':0.9*fontsize}, va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', transform=dic_ax_window[window].transAxes)
                
            # finishing map
            if window == self.windows[-1]:
                gridlines = dic_ax_window[window].gridlines(draw_labels=['bottom', 'left'], zorder=1000, xlabel_style={'fontsize':0.8*fontsize}, ylabel_style={'fontsize':0.8*fontsize} )
            else:
                gridlines = dic_ax_window[window].gridlines(draw_labels=['left'], zorder=1000, xlabel_style={'fontsize':0.8*fontsize}, ylabel_style={'fontsize':0.8*fontsize} )

            # adding colorbar, with size matching axis
            pos0 = dic_ax_window[window].get_position()
            cax = fig.add_axes([pos0.x1+0.010, pos0.y0, 0.01, pos0.height])
            cbar = plt.colorbar(pmesh, cax=cax)
            cbar.ax.tick_params(labelsize=0.8*fontsize)
            cbar.ax.set_xlabel('$\Delta$T ('+u'\u00B0C'+')', size=fontsize)
        return dic_ax_window
    
    def plt_sel( self, ax, fig, obs, ls_event, fontsize ):
        # checking which variable to take
        var = self.var_in_data( obs )
        
        # averaging over region
        average = self.spatial_average( obs[var], self.event_region )
        # selecting window
        dic_value, dic_tm_value = {}, {}
        for window in self.windows:
            if window == 'mean':
                mean_val, calc = self.do_mean( average, yr=self.event_period['Start']['Year'], option_full_outputs_extended=14 )
            
            else:
                mean_val, calc = self.do_rollmean( average, yr=self.event_period['Start']['Year'], window_value=int(window[len('max'):-len('days')]), option_full_outputs=True )
            
            # dealing with bad time formats of some CMIP6 models. This is used to have some good time axis for plot, and occurs only there, not when calculating time serie.
            if type(calc.time.values[0]) != np.datetime64:
                # normally, would do a normal translation of days from this calendar to datetime64
                # however, calendar360days are messing with the whole thing: e.g. February or month31, so iterating over days instead.
                tmp = []
                val_t = calc.time.values[0]
                dict_date = {'Year':val_t.year, 'Month':val_t.month, 'Day':val_t.day - 1}
                for i in np.arange(calc.time.size):
                    # change in day
                    dict_date['Day'] += 1
                    # change in day --> change in month?
                    if dict_date['Day'] > self.n_days(dict_date['Year'], dict_date['Month']):
                        dict_date['Day'] = 1
                        dict_date['Month'] += 1
                        # change in month --> change in year?
                        if dict_date['Month'] > 12:
                            dict_date['Month'] = 1
                            dict_date['Year'] += 1
                    # preparing for datetime64
                    dic_dt = {}
                    for dt in ['Year', 'Month', 'Day']:
                        dic_dt[dt] = str(int(dict_date[dt]))
                        if len(dic_dt[dt]) == 1: dic_dt[dt] = '0'+dic_dt[dt]
                    tmp.append( np.datetime64( '-'.join( [dic_dt['Year'], dic_dt['Month'], dic_dt['Day']] ), 'D') )
                calc.coords['time'] = tmp

            # computing now
            calc = calc.compute()
            
            # calculating over window
            if window == 'mean':
                dic_value[window] = mean_val.values
            else:
                dic_value[window], tm = self.sel_extrema_window( rollmean=mean_val, option_full_outputs=True )
                dic_value[window] = dic_value[window].values
                tm = tm.compute()
                dic_tm_value[window] = calc.time.isel(time=tm).values####################mean_val//calc
                
            dic_value[window] = str(np.round( dic_value[window], 2 ))
            if False:
                # WTH did i want to do there...
                if dic_value[window].size == 1:
                    dic_value[window] = str(np.round( dic_value[window], 2 ))
                else:
                    dic_value[window] = str(np.round( np.mean(dic_value[window]), 2 ))+'+/-'+str(np.round( np.std(dic_value[window]), 2 ))
            
            # members
            if 'member' in calc.coords:
                calc = calc.mean('member')
                mean_val = mean_val.mean('member')
            
            # ploting values
            if window == self.windows[0]:
                calc.plot( label='Daily values of '+self.name_data, color='black', lw=3 )
            if window == 'mean':
                lbl = 'Average over reported period: '
            else:
                lbl = 'Selection on '+window[len('max'):-len('days')]+' days: '
            calc.plot( label=lbl+dic_value[window]+u'\u00B0C', ls='-', lw=2, color=self.colors_windows[self.windows.index(window)] )####################mean_val//calc

        # must select y boundaries after plot of evolutions
        yl = ax.get_ylim()
        
        # ploting selected value
        for window in self.windows:
            if window == 'mean':
                plt.hlines( xmin=self.format_date(self.event_period['Start']), xmax=self.format_date(self.event_period['End']), y=[mean_val.values], color=self.colors_windows[self.windows.index(window)], lw=4, ls=ls_event )
            else:
                plt.vlines( x=[dic_tm_value[window]], ymin=yl[0], ymax=yl[1], color=self.colors_windows[self.windows.index(window)], lw=4, ls=ls_event )
            
        # ploting borders
        val0 = obs[var].time.values[0]
        for tm in self.event_period.keys():
            plt.vlines( x=[self.format_date(dict_date=self.event_period[tm], val0=val0)], ymin=yl[0], ymax=yl[1], color='black', lw=3, ls='--' )#, label=lbl )

        # polishing
        plt.xlim( calc.time.values[0], calc.time.values[-1] )
        plt.ylim( yl[0], yl[-1] )
        plt.xlabel( None )
        plt.grid()
        plt.legend( loc='lower left', prop={'size':0.8*fontsize} )
        # resorting legend
        #handles, labels = plt.gca().get_legend_handles_labels()
        #plt.legend([handles[-1]]+handles[:-1], [labels[-1]]+labels[:-1], loc='lower left', prop={'size':0.7*fontsize})
        
        plt.xticks( size=0.7*fontsize )
        plt.yticks( size=0.8*fontsize )
        
        # ylabel
        plt.ylabel( 'Selection over time', fontsize=0.9*fontsize )
        
        return ax
    
    def plt_tmsr( self, ax, fig, evt_obs, ls_event, fontsize ):
        # looping on windows
        for window in self.windows:
            vals = evt_obs[window].sel(time=self.event_period['Start']['Year']).values
            str_val = str(np.round( vals, 2 ))
            if False:
                # WTH did i want to do there...
                if vals.size == 1:
                    str_val = str(np.round( vals, 2 ))
                else:
                    str_val = str(np.round( np.mean(vals), 2 ))+'+/-'+str(np.round( np.std(vals), 2 ))
            if 'member' in evt_obs[window].coords:
                evt_obs[window].mean('member').plot(ax=ax, label='Time serie using '+window[len('max'):-len('days')]+' days: '+str_val+u'\u00B0C'+' at event', lw=2, color=self.colors_windows[self.windows.index(window)])
            else:
                evt_obs[window].plot(ax=ax, label='Time serie from '+self.name_data+': '+str_val+u'\u00B0C'+' at event', lw=2, color=self.colors_windows[self.windows.index(window)])
            #plt.xlim( 1850, 2018 )
        plt.grid()
        yl = ax.get_ylim()
        plt.vlines( x=[self.event_period['Start']['Year']], ymin=yl[0], ymax=yl[1], color='black', lw=3, ls=ls_event, label='Event' )
        
        # polishing
        plt.ylim( yl )
        plt.xlim( evt_obs[window].time.values[0], evt_obs[window].time.values[-1] )
        plt.xlabel( None )
        plt.ylabel( 'Time serie over the region: $T_{'+window+'}$ ('+ u'\u00B0C'+')', fontsize=0.9*fontsize )
        plt.grid()
        plt.legend( loc=0, prop={'size':0.8*fontsize} )
        plt.xticks( size=0.8*fontsize )
        plt.yticks( size=0.8*fontsize )
        plt.grid()
        return ax
        
    
    def plot_full( self, obs, name_data, path_save, figsize=(15,10), fontsize=15, ls_event=':', close_fig=True ):
        # using the name of the dataset for figures
        self.name_data = name_data
        # select correct months, or all months if period over
        obs_selecttime = self.select_time( obs )

        # preparing figure
        fig = plt.figure( figsize=figsize )
        width_ratios = [1,2]
        height_ratios = list(np.ones(len(self.windows))) + [2]
        spec = gridspec.GridSpec(ncols=2, nrows=len(self.windows)+1, figure=fig, height_ratios=height_ratios, width_ratios=width_ratios, wspace=0.25 )

        # first part: map of region
        dic_ax_window = self.plt_reg( dic_spec_window={window:spec[self.windows.index(window),0] for window in self.windows}, fig=fig, obs_selecttime=obs_selecttime, fontsize=fontsize )

        # second part: selection
        ax = plt.subplot( spec[:len(self.windows),1] )
        ax = self.plt_sel( ax=ax, fig=fig, obs=obs, ls_event=ls_event, fontsize=fontsize )
        pos1 = ax.get_position()
        pos0_top = dic_ax_window[self.windows[0]].get_position()
        pos0_bottom = dic_ax_window[self.windows[-1]].get_position()
        ax.set_position([pos1.x0, pos1.y0 + 0.5 * pos0_bottom.height , pos1.width, pos0_top.y0 + pos0_top.height - pos0_bottom.y0 - 0.5*pos0_top.height])

        # third part: time serie
        evt_obs = self.create_timeserie( obs, name_data )
        ax = plt.subplot( spec[-1,:] )
        ax = self.plt_tmsr( ax=ax, fig=fig, evt_obs=evt_obs, ls_event=ls_event, fontsize=fontsize )
        
        # suptitle
        self.do_suptitle( fig=fig, figsize=figsize, fontsize=fontsize )

        # saving
        fig_name0 = self.name_file_figure_id(path_save, self.windows[0], name_data)
        fig.savefig( os.path.join( path_save, fig_name0 + '.png' ), dpi=300 )
        fig.savefig( os.path.join( path_save, fig_name0 + '.pdf' ) )
        if close_fig:
            plt.close(fig)
        return fig, evt_obs
    # --------------------

    
        
    # --------------------
    # TREATMENT DATA
    def check_event_in_obs( self, data ):
        # checking that it can actually be used for this event:
        t_start, t_end = data.time.values[0], data.time.values[-1]
        return (t_start < self.format_date(dict_date=self.event_period['Start'], val0=t_start)) and (self.format_date(dict_date=self.event_period['End'], val0=t_start) < t_end)
    
    def select_time( self, data ):
        # some events may last longer than a year. Yet, need to cycle through it. averaging over similar periods every time by looping over years
        
        if self.check_event_in_obs( data )== False:
            raise Exception('The period of the event is not contained in the provided dataset')

        # preparing start & end of selection
        t_start, t_end = data.time.values[0], data.time.values[-1]
        if type(t_start) == np.datetime64:
            yr_start, yr_end = pd.DatetimeIndex([t_start]).year[0], pd.DatetimeIndex([t_end]).year[0]
        else:
            yr_start, yr_end = t_start.year, t_end.year

        # looping over all available years in data
        OUT = {}
        for window in self.windows:
            out = []
            for yr in np.arange( yr_start, yr_end+1 ):
                
                if window == 'mean':
                    mean = self.do_mean( data, yr )
                    if (mean is not None):
                        # adding this value
                        out.append( mean.assign_coords( {'time':yr} ) )
                    
                else:
                    # selecting over window & rolling mean
                    rollmean = self.do_rollmean( data, yr, window_value=int(window[len('max'):-len('days')]) )

                    if (rollmean is not None) and (rollmean.time.size > 0):
                        # selecting extrema over window
                        value = self.sel_extrema_window( rollmean=rollmean )

                        # adding this value
                        out.append( value.assign_coords( {'time':yr} ) )

            # concatenating
            OUT[window] = xr.concat( out, dim='time' )#.compute() # not calculating the ensemble, restricting later
        return OUT
    
    
    def do_mean( self, data, yr, option_full_outputs_extended=None ):
        # initial selection
        val0 = data.time.values[0]
        t0 = self.format_date(dict_date={'Year':yr, 'Month':self.event_period['Start']['Month'], 'Day':self.event_period['Start']['Day']}, val0=val0)
        t1 = self.format_date(dict_date={'Year':yr + self.event_period['End']['Year'] - self.event_period['Start']['Year'], 'Month':self.event_period['End']['Month'], 'Day':self.event_period['End']['Day']}, val0=val0 )

        # average
        calc = data.sel( time=slice(t0,t1) )
        if calc.time.size == 0:
            return None
        else:
            mean = calc.mean('time')

            if option_full_outputs_extended is not None:
                # widening selection
                if type( val0 ) == np.datetime64:
                    t0_tmp = pd.DatetimeIndex([t0]).shift(periods=-7, freq='D')[0]
                    t1_tmp = pd.DatetimeIndex([t1]).shift(periods=7, freq='D')[0]
                else:
                    t0_tmp = xr.CFTimeIndex( [t0] ).shift( freq='D', n=-option_full_outputs_extended )[0]
                    t1_tmp = xr.CFTimeIndex( [t1] ).shift( freq='D', n=option_full_outputs_extended )[0]
                t0_large = self.format_date(dict_date={'Year':t0_tmp.year, 'Month':t0_tmp.month, 'Day':t0_tmp.day}, val0=val0)
                t1_large = self.format_date(dict_date={'Year':t1_tmp.year, 'Month':t1_tmp.month, 'Day':t1_tmp.day}, val0=val0)
                calc_large = data.sel( time=slice(t0_large,t1_large) )
                return mean, calc_large
            else:
                return mean
        

    def do_rollmean( self, data, yr, window_value, option_full_outputs=False ):
        # initial selection
        val0 = data.time.values[0]
        t0 = self.format_date(dict_date={'Year':yr, 'Month':self.event_period['Start']['Month'], 'Day':self.event_period['Start']['Day']}, val0=val0)
        t1 = self.format_date(dict_date={'Year':yr + self.event_period['End']['Year'] - self.event_period['Start']['Year'], 'Month':self.event_period['End']['Month'], 'Day':self.event_period['End']['Day']}, val0=val0 )

        # widening selection
        if type( val0 ) == np.datetime64:
            t0_tmp = pd.DatetimeIndex([t0]).shift(periods=-window_value, freq='D')[0]
            t1_tmp = pd.DatetimeIndex([t1]).shift(periods=window_value, freq='D')[0]
        else:
            t0_tmp = xr.CFTimeIndex( [t0] ).shift( freq='D', n=-window_value )[0]
            t1_tmp = xr.CFTimeIndex( [t1] ).shift( freq='D', n=window_value )[0]
        t0_large = self.format_date(dict_date={'Year':t0_tmp.year, 'Month':t0_tmp.month, 'Day':t0_tmp.day}, val0=val0)
        t1_large = self.format_date(dict_date={'Year':t1_tmp.year, 'Month':t1_tmp.month, 'Day':t1_tmp.day}, val0=val0)
            

        # rolling average
        calc = data.sel( time=slice(t0_large,t1_large) )
        if calc.time.size == 0:
            return None
        else:
            rollmean = calc.rolling(time=window_value, center=True).mean()

            # cutting rollmean to initial selection, to avoid out-of-window selection
            rollmean = rollmean.sel( time=slice(t0, t1) )

            if option_full_outputs:
                return rollmean, calc
            else:
                return rollmean


    def sel_extrema_window( self, rollmean, option_full_outputs=False ):
        if self.event_type[2] == 'min':
            value = rollmean.min('time', skipna=True)
            if option_full_outputs:
                tm_value = rollmean.argmin('time', skipna=True)
        else:
            value = rollmean.max('time', skipna=True)
            if option_full_outputs:
                tm_value = rollmean.argmax('time', skipna=True)
            
        if option_full_outputs:
            return value, tm_value
        else:
            return value

        
    def var_in_data( self, ds ):
        # event type: which variable to use?
        if self.evt['Disaster Type'].values in ['Extreme temperature ', 'Extreme temperature', ' Extreme temperature']:
            var = 'temperature'
        else:
            raise Exception("Prepare this classification")
        
        # identifying variable in dataset
        if var == 'temperature':
            vars_in_data = [vv for vv in ['temperature', 'tas', 't2m'] if vv in ds]
        else:
            raise Exception("Variable not prepared")
            
        # checks on variable
        if len( vars_in_data ) > 1:
            raise Exception("Too many potential variables found in dataset.")
        elif len(vars_in_data) == 0:
            raise Exception("No corresponding variables found in dataset.")
        return vars_in_data[0]
            
    @staticmethod
    def mask_reg( event_region, lon, lat ):
        reg = regionmask.Regions([event_region.item()])
        mask = mask_3D_frac_approx(reg, lon, lat)
        if mask.region.size==0:
            # problem: mask is 0, probably because this region is too small
            # solution: taking the one grid cell over the centroids of the polygons --> may cause issues for sparsely small regions (eg small cities, islands)
            global_centroid = np.mean( reg.centroids, axis=0 )
            lon_c, lat_c = lon.values[np.argmin( np.abs(global_centroid[0] - lon.values) )], lat.values[np.argmin( np.abs(global_centroid[1] - lat.values) )]
            mask = xr.zeros_like( lat * lon )
            mask.loc[{'lon':lon_c, 'lat':lat_c}] = 1
            return mask
            
        else:
            return mask.sel(region=0).drop( ('region', 'abbrevs', 'names') )
    
    def spatial_average( self, data_temp, event_region ):
        # some ESMs have their lat & lon not equally spaced (yeah yeah...), so averaging spacing
        df_lon, df_lat = np.diff(data_temp['lon'].values), np.diff(data_temp['lat'].values)
        if len(set(df_lon)) > 1:
            data_temp.coords['lon'] = data_temp['lon'].values[0] + np.mean(df_lon) * np.arange(data_temp['lon'].size)
        if len(set(df_lat)) > 1:
            data_temp.coords['lat'] = data_temp['lat'].values[0] + np.mean(df_lat) * np.arange(data_temp['lat'].size)
        mask = self.mask_reg( event_region, data_temp.lon, data_temp.lat )
        if mask is None:
            return None
        else:
            return data_temp.weighted(mask).mean( ('lat','lon') )
    
    def spatial_quantile( self, q, data_temp, event_region, var_data ):
        mask = self.mask_reg( event_region, data_temp.lon, data_temp.lat )
        if mask is None:
            return None
        else:
            dm = DescrStatsW( data=data_temp[var_data].values.flatten(), weights=mask.values.flatten() )
            return  dm.quantile( probs=q ).values[0]
        
    def create_timeserie( self, data, name_data, option_select_spa1st=False ):
        # identifying correct variable
        var = self.var_in_data( data )
        
        # initialize
        output = {}
        if option_select_spa1st: # spatial average THEN selection on time
            # calculating average
            average = self.spatial_average( data[var], self.event_region )
            
            # select correct months, or all months if period over 
            data_selecttime = self.select_time( average )
            
            # compute
            for window in self.windows:
                output[window] = data_selecttime[window].compute()

            
        else: # selection on time THEN spatial average ---> makes more sense from how it was felt/perceived by people, as a driver for people
            # select correct months, or all months if period over 
            self.data_selecttime = self.select_time( data )

            output = {}
            for window in self.windows:
                # calculating average
                average = self.spatial_average( self.data_selecttime[window][var], self.event_region )

                # compute
                output[window] = average.compute()
        if self.option_detailed_prints:
            print("Timeseries for "+name_data+" calculated")
        return output
    # --------------------
    
    
    # --------------------
    def name_file_timeseries(self, path_save, window, name_data):
        # checking path
        path_save = os.path.join(path_save, self.identifier_event)
        if not os.path.exists(path_save):os.makedirs(path_save)
            
        # name of file
        basis = os.path.join( path_save, 'timeserie_' + str(self.identifier_event) )
        info_data = name_data + '-' + window
        return basis + '_' + info_data + '.nc'
    
    def name_file_figure_id(self, path_save, window, name_data):
        # checking path
        path_save = os.path.join(path_save, self.identifier_event)
        if not os.path.exists(path_save):os.makedirs(path_save)
            
        # name of file
        basis = os.path.join( path_save, 'figure_' + str(self.identifier_event) )
        info_data = name_data + '-' + window
        return basis + '_' + info_data + '_id'
    
    def save_timeserie(self, data_tsr, name_data, path_save):
        for window in data_tsr.keys():
            # preparing dataset
            OUT = xr.Dataset()
            OUT[window] = data_tsr[window]
            OUT[window].attrs['unit'] = 'C'
            OUT.attrs['Type'] = self.event_type[1]
            OUT.attrs['Country'] = self.country
            OUT.attrs['Location'] = self.location_init
            OUT.attrs['Start'] = str( self.event_period['Start'] )
            OUT.attrs['End'] = str( self.event_period['End'] )
            OUT.attrs['name_data'] = name_data
            
            # saving dataset
            OUT.to_netcdf( self.name_file_timeseries(path_save, window, name_data), encoding={var: {"zlib": True} for var in OUT.variables} )
            
    def test_load_all(self, list_data_to_do, path_save ):
        return np.all( [os.path.isfile( self.name_file_timeseries(path_save, window, name_data) ) for name_data in list_data_to_do for window in self.windows] )
            
    def load_timeseries(self, name_data, path_save):
        out = {}
        for window in self.windows:
            tsr = xr.open_dataset( self.name_file_timeseries(path_save, window, name_data) )
            out[window] = tsr[window]
        return out
    # --------------------
#---------------------------------------------------------------------------------------------------------------------------

