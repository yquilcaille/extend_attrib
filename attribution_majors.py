import sys
import time as ti

import numpy as np
import xarray as xr

from fcts_support_basic import *
from fcts_support_event import *
from fcts_support_io import *
from fcts_support_plot_v4 import *
from fcts_support_synthesis import *
from fcts_support_training import *

#==================================================================================================
# 0. OPTIONS
#==================================================================================================
# Options for selection of event (EM-DAT)
disasters = ['Heat wave'] #['Extreme temperature '] #+ ['Drought', 'Wildfire'] + ['Flood'] + ['Storm'] + ['Fog', 'Glacial lake outburst', 'Landslide', 'Oceans']
emdat_start_year = 2000
emdat_end_year = 2022
threshold_SequenceMatcher = 0.85

# Options for training of events
reference = 'ERA5' # this dataset will be the one used to select the best fit and the probability level of each event for other datasets
potential_distributions = ['GEV']# 'GPD', 'skewnorm', 'gaussian'
# within 'constant', 'linear', 'power2', 'poly2', 'power3', 'poly3', 'sigmoids'
potential_evolutions = {'loc':  ['linear'],\
                        'scale':['constant'],\
                        'shape':['constant'],\
                        'mu':   ['constant', 'linear', 'power2', 'poly2'],\
                       }
select_BIC_or_NLL = 'BIC'       # BIC, NLL   (default BIC)   : /!\ FORMER axis of sensitivity analysis --> observed that linear_constant_constant is good most of the time (99% of the improvement in BIC is done with linear)
training_start_year_obs = 1950  # 1920, 1950 (default 1950)  : /!\ FORMER axis of sensitivity analysis when only BEST was used. Now, using ERA5 as reference.
training_end_year_CMIP6 = 2100  # 2022, 2100 (default 2100)  : /!\ Axis of sensitivity analysis
option_train_wo_event = False   # True, False (default False): /!\ Axis of sensitivity analysis  
weighted_NLL = True             # True, False (default True) : /!\ Axis of sensitivity analysis
xtol_req = 1.e-9

# Options for figures & results
option_load_precalc = ['timeserie', 'parameters', 'synthesis'] #['timeserie', 'parameters', 'synthesis'] # if not an empty list, will try to load them. Warning, will prevent plots of time series / tree_results already produced.
option_save = True # if True, will save time series & fitted parameters.
option_plot = [] # 'timeserie', 'distributions', 'tree'

# Options for carbon majors
level_OSCAR = 'mean'

# Options for synthesis
add_text_files_synthesis = None

# Options for CMIP6 data
n_ESMs_CMIP6 = 10 # /!\ Axis of sensitivity analysis
option_run_CMIP6 = True
list_experiments_CMIP6 = ['historical','ssp245']
list_members_CMIP6 = ['r1i1p1f1']
period_comparison_seasons = [1950, 2020]

# Options for run on server / else
run_on_server = False
option_detailed_prints = False # if True, will print extensive messages. Warning, may overload the console

# Paths related to data
paths_in = {'BEST':'/net/exo/landclim/data/dataset/Berkeley-Earth-Data_gridded/20141204/1deg_lat-lon_1d/original',\
            'CMIP6':'/net/atmos/data/cmip6',\
            'CMIP6-ng':'/net/ch4/data/cmip6-Next_Generation',\
            'EMDAT': '/net/exo/landclim/yquilcaille/contributions_FF/EMDAT',\
            'ERA5':'/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_1h/processed/regrid_tmean1d',\
            'FF': '/net/exo/landclim/yquilcaille/contributions_FF/FF',\
            'GCB': '/net/exo/landclim/yquilcaille/contributions_FF/GCB',\
            'GeoBoundaries': '/net/exo/landclim/yquilcaille/contributions_FF/GeoBoundaries',\
            'GMT': '/net/exo/landclim/yquilcaille/contributions_FF/GMT',\
            'exclusions_CMIP6':'/net/exo/landclim/yquilcaille/contributions_FF/ancillary_data/exclusions_CMIP6',\
            'select_CMIP6':'/net/exo/landclim/yquilcaille/contributions_FF/select_CMIP6',\
            'regions_WB':'/net/exo/landclim/yquilcaille/contributions_FF/data_regions',\
            'Jones2023':'/net/exo/landclim/yquilcaille/contributions_FF/Jones023'
           }
paths_out = {'results': '/net/exo/landclim/yquilcaille/contributions_FF/results',\
             'figures': '/home/yquilcaille/contributions_FF/figures'}
#==================================================================================================
#==================================================================================================







#==================================================================================================
# 1. GENERAL PREPARATION
#==================================================================================================
# 1.1. Preparation of runs
if run_on_server:
    os.nice(19)
    nb_csl = int(sys.argv[1])
    index_csl = int(sys.argv[2])
    print( "Running job #"+str(index_csl)+' out of '+str(nb_csl) )
else:
    print('Running whole job')
    
# 1.2. Preparation of some repositories
for tmp in ['timeseries', 'training', 'synthesis']:
    paths_out['results_'+tmp] = os.path.join( paths_out['results'], tmp )
    paths_out['figures_'+tmp] = os.path.join( paths_out['figures'], tmp )
for path in paths_out.values():
    if not os.path.exists(path):os.makedirs(path)    
#==================================================================================================
#==================================================================================================





#==================================================================================================
# 2. PREPARING DATA
#==================================================================================================
# 2.1. Preparing EM-DAT
emdat, emdat_removed_years = func_prepare_emdat( path=paths_in['EMDAT'], disasters=disasters, start_year=emdat_start_year, end_year=emdat_end_year, option_detailed_prints=option_detailed_prints )

# 2.2. Preparing geographical boundaries
geobounds, dict_geobounds = func_prepare_geobounds( path=paths_in['GeoBoundaries'], source='gadm', option_detailed_prints=option_detailed_prints )

# 2.3 Preparing observations: BEST (Important, missing data after 31.07.2022 => cannot attribute 2022 heatwaves using BEST)
best, gmt_best = func_prepare_BEST( path=paths_in['BEST'], path_gmt=paths_in['GMT'], option_detailed_prints=option_detailed_prints )
data_to_do = {'BEST':[best['tavg'], gmt_best]}

# 2.4 Preparing observations: ERA5-Land
era5, gmt_era5 = func_prepare_ERA5( path=paths_in['ERA5'], path_gmt=paths_in['GMT'], option_detailed_prints=option_detailed_prints )
data_to_do['ERA5'] = [era5['t2m'], gmt_era5]

# 2.5. Preparing CMIP6 data
var_input_CMIP6 = 'tas'
if option_run_CMIP6:
    if False:
        # Preparing CMIP6 files
        if len( [xp for xp in list_experiments_CMIP6 if 'ssp' in xp] ) > 2:
            raise Exception("preprocess and treatment of CMIP6 files has been designed to run with 1 historical and 1 ssp, not more.")
        cmip6 = files_cmip6(var_input=var_input_CMIP6, list_experiments=list_experiments_CMIP6, list_members=list_members_CMIP6, \
                            path_cmip6=paths_in['CMIP6'], path_exclusions_cmip6=paths_in['exclusions_CMIP6'], option_detailed_prints=option_detailed_prints)
        cmip6.gather_files(forced_domain=['day'], option_single_grid=True)
        cmip6.filter_runs()
        cmip6.load_all(path_gmt=paths_in['GMT'])
    else:
        # Preparing CMIP6-ng files
        cmip6 = files_cmip6ng(var_input=var_input_CMIP6, list_experiments=list_experiments_CMIP6, list_members=list_members_CMIP6, path_cmip6ng=paths_in['CMIP6-ng'], option_detailed_prints=option_detailed_prints)
        # Loading CMIP6-ng
        cmip6.load_all(path_gmt=paths_in['GMT'])
    
    # preparing that in data to run:
    for esm in cmip6.data_esm.keys():
        data_to_do[esm] = [cmip6.data_esm[esm], cmip6.gmt_esm[esm]]

# preparing list of datasets to do, putting reference first
datasets = [reference] + [key for key in data_to_do.keys() if key!=reference]

# 2.6. Selection of CMIP6 data
comp_cmip6ng = select_cmip6( path=paths_in['select_CMIP6'], period_seasons=period_comparison_seasons )

# 2.7. Preparing OSCAR data
dGMT_OSCAR = func_prepare_OSCAR_GMT( path_gmt=paths_in['GMT'], option_detailed_prints=option_detailed_prints )

# 2.8. Preparing FF emissions database
emissions_FF = xr.open_dataset( os.path.join(paths_in['FF'], 'emissions_majors_1850-2022.nc') )

# 2.9. Preparing GCB database
emissions_GCB = func_prepare_GCB( path_gcb=paths_in['GCB'] )

# 2.10. Preparing regions database for synthesis
dico_ISO2country, dico_country2reg = func_prepare_regions( path_reg=paths_in['regions_WB'] )

# 2.11. Preparing Jones2023 CH4 database
emissions_Jones023 = xr.open_dataset( os.path.join(paths_in['Jones2023'], 'emissions_Jones2023-CH4_1830-2021.nc') )

print('All supporting data loaded.')
print('')# one line for better readibility
#==================================================================================================
#==================================================================================================








#==================================================================================================
# 3. ANALYZING EVENT, TRAINING DISTRIBUTIONS, EVALUATING CONTRIBUTIONS
#==================================================================================================
if run_on_server:
    indexes_events = emdat.index.values[ np.arange( int(index_csl*emdat.index.size/nb_csl), np.min([emdat.index.size, int((index_csl+1)*emdat.index.size/nb_csl) + (index_csl==nb_csl-1)]) ) ]
else:
    indexes_events = emdat.index.values

results = {}
for i, ind_evt in enumerate(indexes_events):
    disno = str(emdat['DisNo.'].sel(index=ind_evt).values)
    print('Treating event: ' + str(disno) + ' (' + str(i+1) + '/' + str(len(indexes_events)) + ')' )
    time0 = ti.time()

    #------------------------------------------------------------------------------------------
    # 3.1. PREPARING EVENT
    evt_obs = {}

    # defining event
    evt_id = treat_event( evt=emdat.sel(index=ind_evt), geobounds=geobounds, dict_geobounds=dict_geobounds, threshold_SequenceMatcher=threshold_SequenceMatcher, option_detailed_prints=option_detailed_prints )
    
    # identifying ensemble of spatial units within the ISO of interest: need to define region before selecting ESMs
    evt_id.def_spatial_units()

    # selection for CMIP6 if necessary
    if option_run_CMIP6:
        evt_id.select_ESMs_CMIP6( comp_cmip6=comp_cmip6ng, n_ESMs_CMIP6=n_ESMs_CMIP6 )
    # will produce ALL CMIP6 ESMs, not just the pre-selection. (alternative: 'datasets[:2] + evt_id.kept_esms', however hinders sensitivity analysis on n_ESMs_CMIP6)
    datasets_evt = datasets
        
    # taking only valid datasets, ie if event is represented by observations
    datasets_evt = [name_data for name_data in datasets_evt if evt_id.check_event_in_obs( data=data_to_do[name_data][0] )]
    
    # testing whether need to calculate all required time series
    if ('timeserie' in option_load_precalc) and evt_id.test_load_all(list_data_to_do=datasets_evt, path_save=paths_out['results_timeseries'] ):
        # loading them all
        if option_detailed_prints:
            print('Loading existing time series of event')
        for name_data in datasets_evt:
            evt_obs[name_data] = evt_id.load_timeseries(name_data=name_data, path_save=paths_out['results_timeseries'])

    else:
        # calculating time serie, and if necessary, doing figure
        if 'timeserie' in option_plot:
            # plot & time series
            for name_data in datasets_evt:
                if evt_id.check_event_in_obs( data=data_to_do[name_data][0] ):
                    fig_id, evt_obs[name_data] = evt_id.plot_full( obs=data_to_do[name_data][0], path_save=paths_out['figures_timeseries'], name_data=name_data, close_fig=True )
                    
                    # saving for later if needed
                    if option_save:
                        evt_id.save_timeserie( data_tsr=evt_obs[name_data], name_data=name_data, path_save=paths_out['results_timeseries'] )

        else:
            # spatial & temporal average in region of interest for observation
            evt_obs = {}
            for name_data in datasets_evt:
                if evt_id.check_event_in_obs( data=data_to_do[name_data][0] ):
                    if option_detailed_prints:
                        print('timeseries with ' + name_data)
                    evt_obs[name_data] = evt_id.create_timeserie( data_to_do[name_data][0], name_data=name_data )

                    # saving for later if needed
                    if option_save:
                        evt_id.save_timeserie( data_tsr=evt_obs[name_data], name_data=name_data, path_save=paths_out['results_timeseries'] )
    #------------------------------------------------------------------------------------------
    
    #------------------------------------------------------------------------------------------
    # 3.2. TRAINING ON LARGE ENSEMBLE OF CONFIGURATIONS
    # keeping track of trained distributions for analysis on each of them, and to use reference for other datasets
    evt_fits = {}
    for name_data in datasets_evt:
        if evt_id.check_event_in_obs( data=data_to_do[name_data][0] ):
            # preparing period of fit
            training_start_year = {False:1850, True:training_start_year_obs}[name_data in ['BEST', 'ERA5']]
            training_end_year = {False:2022, True:training_end_year_CMIP6}[name_data not in ['BEST', 'ERA5']]

            # preparing evt_fit
            evt_fits[name_data] = train_distribs(data_gmt=data_to_do[name_data][1], data_obs=evt_obs[name_data], event_year=evt_id.event_year, identifier_event=evt_id.identifier_event,\
                                                 name_data=name_data, name_reference=reference, training_start_year=training_start_year, training_end_year=training_end_year, \
                                                 potential_evolutions=potential_evolutions, potential_distributions=potential_distributions,xtol_req=xtol_req,\
                                                 weighted_NLL=weighted_NLL, select_BIC_or_NLL=select_BIC_or_NLL, option_train_wo_event=option_train_wo_event, n_iterations_BS=1000,\
                                                 path_results=paths_out['results_training'], path_figures=paths_out['figures_training'], option_detailed_prints=option_detailed_prints)

            if name_data != reference:
                # using calculations on reference to restrain the set of configurations & pass probability levels at time of event
                evt_fits[name_data].learn_from_ref( ref=evt_fits[reference] )

            if ('parameters' in option_load_precalc) and evt_fits[name_data].test_load_all():
                # loading them all
                if option_detailed_prints:
                    print('Loading existing parameters for event with '+name_data)
                evt_fits[name_data].load_parameters()

            else:
                # train a set of configurations (large on reference, 1 otherwise)
                evt_fits[name_data].fit_all()

                # select best one, based on its BIC  or  NLL
                evt_fits[name_data].select_best_fit()

                # bootstrapping the best solution
                evt_fits[name_data].bootstrap_best_fit()# with 1000, average seems to be varying by 1% with tests

                # saving for later if needed
                if option_save:
                    evt_fits[name_data].save_parameters()

            # evolution of parameters for bootstrapped solutions: factual
            evt_fits[name_data].calc_params_bootstrap( predictor=data_to_do[name_data][1], label='with_CC' )

            # evolution of parameters for bootstrapped solutions: counterfactual
            if 1850 in data_to_do[name_data][1].time:
                PI = data_to_do[name_data][1].sel(time=slice(1850,1900)).mean()
            else:
                PI = data_to_do[name_data][1].sel(time=slice(1961,1990)).mean() - 0.36 # IPCC AR6 Chapter 2, Cross Chapter Box 2.3, Table 1
            evt_fits[name_data].calc_params_bootstrap( predictor = PI * xr.ones_like( data_to_do[name_data][1]), label='without_CC' )

            # probas_with_CC, probas_without_CC, attrib_metric
            evt_fits[name_data].full_calc_probas( evt_obs=evt_obs[name_data] )

            # plot
            if 'distributions' in option_plot:
                fig_fit = evt_fits[name_data].plot_full( plot_start_year=1950, close_fig=True )

                if ('tree' in option_plot) and evt_fits[name_data].plot_trees:
                    for window in evt_fits[name_data].list_windows:
                        # prepare tree
                        tree = tree_results_fit( results_fit=evt_fits[name_data].results_fits[window], do_empty_nodes=True )

                        # prepare positions
                        tree.calculate_positions_nodes(layout="rt_circular") # 'kk', 'auto', 'rt_circular', 'fr', 'rt'
                        # plot tree
                        fig = tree.plot(figsize=(1000, 1000), sizes={"dots": 25, "configuration": 15, "distribution": 15, "expression": 8, "BIC": 8, 'empty':1},\
                                        colors={ "lines": "rgb(200,200,200)", "nodes": "rgb(100,100,100)", "edges": "rgb(100,100,100)", "background": "rgb(248,248,248)", "text": "rgb(0,0,0)" },\
                                        name_save=evt_fits[name_data].name_file_figure_tree(window) )
    #------------------------------------------------------------------------------------------
    
    #------------------------------------------------------------------------------------------
    # 3.3. DEDUCING FOR EACH CARBON MAJOR
    synth = synthesis( evt_id, evt_obs, evt_fits, data_to_do, training_start_year, training_end_year, option_detailed_prints=option_detailed_prints )
    synth.select_datasets(comp_cmip6ng=comp_cmip6ng, n_ESMs_CMIP6=n_ESMs_CMIP6)
    synth.synthesis_global()
    if ('synthesis' in option_load_precalc) and synth.test_load_all( path_save=paths_out['results_synthesis'], add_text=add_text_files_synthesis, level_OSCAR=level_OSCAR ):
        synth.load_synthesis( path_save=paths_out['results_synthesis'], add_text=add_text_files_synthesis, level_OSCAR=level_OSCAR )

    else:
        synth.synthesis_entities( dGMT_OSCAR, emissions_FF, emissions_GCB, level_OSCAR=level_OSCAR )
        if option_save:
            synth.save_synthesis( path_save=paths_out['results_synthesis'], add_text=add_text_files_synthesis)
    #------------------------------------------------------------------------------------------

    if run_on_server==False:
        results[disno] = [ evt_id, datasets_evt, evt_obs, evt_fits, synth ]

    if option_detailed_prints:
        print('Finished with this event in: '+str(int( (ti.time()-time0)/60 ))+'min')
print('Finished!')
#==================================================================================================
#==================================================================================================










#==================================================================================================
# 4. PANORAMA (ie synthesis on all events & carbon majors)
#==================================================================================================
if False:
    pano = panorama(emissions_FF, dico_ISO2country, dico_country2reg,\
                    n_ESMs_CMIP6, reference, level_OSCAR, training_start_year_obs, training_end_year_CMIP6, option_train_wo_event, select_BIC_or_NLL, weighted_NLL)
    if pano.test_load(path_save=paths_out['results_synthesis'], add_text=add_text_files_synthesis):
        pano.load_synthesis(path_save=paths_out['results_synthesis'], add_text=add_text_files_synthesis)
        pano.eval_ISOs(eval_ISOs_from_results=False, emdat=emdat, indexes_events=indexes_events, geobounds=geobounds, dict_geobounds=dict_geobounds, threshold_SequenceMatcher=threshold_SequenceMatcher)
    else:
        pano.gather_values(results)
        pano.eval_ISOs(eval_ISOs_from_results=True)
        pano.save_synthesis(path_save=paths_out['results_synthesis'], add_text=add_text_files_synthesis)
    pano.calc()
#==================================================================================================
#==================================================================================================














#==================================================================================================
# 5. FIGURES & TABLES
#==================================================================================================
#------------------------------------------------------------------------------------------
# 5.4. FOURTH VERSION OF THE FIGURES
#------------------------------------------------------------------------------------------
if False:
    # Figure 1 -------> put some of 2022
    tmp_fig = figure1_v4( results=results, ind_evts=['2021-0390-USA', '2003-0391-FRA', '2022-0248-IND', '2013-0582-CHN'], emdat=emdat, data_to_do=data_to_do, name_data='ERA5')# alternatives India: '2010-0206-IND'
    fig = tmp_fig.plot(path_save=paths_out['figures_synthesis'])
    
    # Figure 2
    tmp_fig = figure2_v4(emdat=emdat, pano=pano)
    fig = tmp_fig.plot(path_save=paths_out['figures_synthesis'])

    # Figure 3
    tmp_fig = figure3_v4(emissions_FF=emissions_FF, emissions_GCB=emissions_GCB, dGMT_OSCAR=dGMT_OSCAR, data_to_do=data_to_do, emissions_Jones023=emissions_Jones023, emdat=emdat,\
                          entities_plot=['Tier1', 'Abu Dhabi National Oil Company', 'Petrobras'], pano=pano, method_entity='GMT')#'Shell', 
    fig = tmp_fig.plot(path_save=paths_out['figures_synthesis'])

    # Table
    tmp_tab = create_table_all(emdat, pano, emissions_FF, results )
    tmp_tab.create(path_save=paths_out['figures_synthesis'])
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#==================================================================================================
#==================================================================================================













