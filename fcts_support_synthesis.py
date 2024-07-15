from copy import deepcopy

import cartopy.crs as ccrs
import numpy as np
import xarray as xr

from fcts_support_basic import *
from fcts_support_event import *
from fcts_support_training import *


#---------------------------------------------------------------------------------------------------------------------------
# CLASS SYNTHESIS ON DATASETS & ENTITIES, for ONE event
#---------------------------------------------------------------------------------------------------------------------------
class synthesis:
    #--------------------------------------------------------------
    # BASIC FUNCTIONS
    def __init__( self, evt_id, evt_obs, evt_fits, data_to_do, training_start_year, training_end_year, option_detailed_prints=False ):
        self.evt_id = evt_id
        self.evt_obs = evt_obs
        self.evt_fits = evt_fits
        self.data_to_do = data_to_do
        self.training_start_year = training_start_year
        self.training_end_year = training_end_year
        self.option_detailed_prints = option_detailed_prints
        self.obs_products = ['ERA5', 'BEST']
        self.identifier_event = self.evt_id.identifier_event
        
        # immediately testing if multiple windows. was kept as a feature, but is meant to be removed. for now, checking if only one.
        if len(evt_id.windows) > 1:
            raise Exception("Multiple windows")
        else:
            self.window = evt_id.windows[0]
            
        
    def select_datasets( self, n_ESMs_CMIP6=None, comp_cmip6ng=None ):
        # all available datasets
        self.selection0 = list(self.evt_obs.keys())
        
        # will order all datasets by their performances in reproducing the seasonality over region
        # then select in them those that respect the set of parameters
        # if not reaching the set number of ESMs, take the best ones according to seasonality but not necessarily respecting paramets

        # ordering all datasets
        self.selection1 = self.selection_seasonality(n_ESMs_CMIP6=comp_cmip6ng.esm.size, comp_cmip6ng=comp_cmip6ng, datasets_start=self.selection0)
        
        # removing those that dont respect the parameters
        self.selection2 = self.selection_parameters(datasets_start=self.selection1)
        self.select_obs = [dt for dt in self.selection2 if dt in self.obs_products]
        self.select_mod = [dt for dt in self.selection2 if dt not in self.obs_products]

        # complete if not enough:
        self.n_ESMs_CMIP6 = n_ESMs_CMIP6
        if len(self.select_mod) >= self.n_ESMs_CMIP6:# should have an axis to mark datasets as observations or models, would avoid having this hard coded 2 obs.
            # good selection: respect parameters, and 
            self.datasets_postselection = self.select_obs + self.select_mod[:self.n_ESMs_CMIP6]
        else:
            # issue with selection: not enough satisfying parameters, completing with best ones among remaining ones with good seasonality
            self.datasets_postselection, n_added = self.selection2.copy(), 0
            for d in self.selection1:
                if (d not in self.datasets_postselection) and (len(self.select_mod) + n_added < self.n_ESMs_CMIP6):
                    self.datasets_postselection.append( d )
                    n_added += 1
            
        if self.option_detailed_prints:
            print("Selection of datasets: initial=" + str(len(self.evt_obs.keys())) + "; pre-selection: " + str(len(self.selection1)) + "; post-selection: " + str(len(self.datasets_postselection)))
    #--------------------------------------------------------------

    
    #--------------------------------------------------------------
    # SELECTION
    def selection_seasonality( self, n_ESMs_CMIP6, comp_cmip6ng, datasets_start ):
        if (n_ESMs_CMIP6 is None) and (comp_cmip6ng is None):
            # at the moment take them all.
            datasets_out = datasets_start
        else:
            # selecting
            obs = [dt for dt in datasets_start if dt not in comp_cmip6ng.esm]
            mod = [dt for dt in datasets_start if dt in comp_cmip6ng.esm]
            
            # taking n best ESMs according to climatology
            self.evt_id.select_ESMs_CMIP6( comp_cmip6=comp_cmip6ng.sel(esm=mod), n_ESMs_CMIP6=n_ESMs_CMIP6 )
            datasets_out = obs + self.evt_id.kept_esms
        return datasets_out
        
    def selection_parameters( self, datasets_start ):
        # preparation
        tmp = list(self.evt_fits.keys())[0]
        self.ref = self.evt_fits[tmp].name_reference
        self.conf_interval = self.evt_fits[self.ref].conf_interval
        self.dico_q_confid = self.evt_fits[self.ref].dico_q_confid

        # calculating limits in scale of reference dataset
        limits_ref = {}
        for p in ['loc', 'scale', 'shape']:
            if p in self.evt_fits[self.ref].parameters[self.window]:
                limits_ref[p] = self.evt_fits[self.ref].parameters[self.window][p].sel(label='with_CC', time=self.evt_fits[self.ref].event_year).quantile(q=list(self.dico_q_confid.values()), dim='bootstrap').drop('quantile').values

        # checking for each ESM whether works: similarity of scale with ref, but also checking whether PR >= XXX (diverge)
        datasets_out, datasets_rejected = [],[]
        for name_data in datasets_start:
            # calculating limits of dataset
            limits = {}
            for p in ['loc', 'scale', 'shape']:
                if p in self.evt_fits[name_data].parameters[self.window]:
                    limits[p] = self.evt_fits[name_data].parameters[self.window][p].sel(label='with_CC', time=self.evt_fits[name_data].event_year).quantile(q=list(self.dico_q_confid.values()), dim='bootstrap').drop('quantile').values
            PR_median = self.evt_fits[name_data].PR[self.window]['median']
            # comparing limits of dataset to reference
            test = {}
            for p in limits_ref:
                test[p] = (limits[p][1] < limits_ref[p][0])  or  (limits_ref[p][1] < limits[p][0])
            if test['scale'] or test['shape']:# or (PR_median >= XXXX):# not location, because base level depend on dataset
                datasets_rejected.append( name_data )
            else:
                datasets_out.append( name_data )
        return datasets_out#, datasets_rejected
    #--------------------------------------------------------------

    
    
    
    
    #--------------------------------------------------------------
    # PROBABILITY RATIO & INTENSITIES
    def test_systematic_differences( self, values ):
        stat_test = ss.chisquare( f_obs=np.hstack(values) ).statistic / (len(self.datasets_postselection) - 1)
        if stat_test > 1:
            self.WARNING_systematic_differences = "WARNING: Systematic differences among models contribute significantly! Propagating uncertainty through sqrt(chi2/dof)!"
            if self.option_detailed_prints:
                print(self.WARNING_systematic_differences)
            self.factor_model_uncertainty = np.sqrt( stat_test )
        else:
            self.WARNING_systematic_differences = None
            self.factor_model_uncertainty = 1
    
    
    def func_synthesis( self, dict_values, method_synthesis='equality_obs_models' ):
        """
            method_synthesis:
                - equality_datasets: average together all observations and models.
                - equality_obs_models: averaging first all observations, then all models, then average.
        """
        # pooling together probability ratios
        sth = xr.Dataset()

        for pri in ['PR', 'I']:
            nmdt0 = list(dict_values.keys())[0]
            if 'entity' not in dict_values[nmdt0][pri+'_values'].coords:
                sth[pri+'_values'] = xr.DataArray(np.nan, dims=('datasets', 'bootstrap',),\
                                                  coords={'datasets':self.datasets_postselection, 'bootstrap':dict_values[self.ref].bootstrap})
            else:
                sth[pri+'_values'] = xr.DataArray(np.nan, dims=('datasets', 'bootstrap', 'entity',),\
                                                  coords={'datasets':self.datasets_postselection, 'bootstrap':dict_values[self.ref].bootstrap, 'entity':dict_values[self.ref].entity.values})
            for name_data in self.datasets_postselection:
                sth[pri+'_values'].loc[{'datasets':name_data}] = dict_values[name_data][pri+'_values'].values
            
            # blocking very high PR cf WWA approach: only for analysis, plots & interpretation, but not during calculations
            #vals = xr.where( (sth['values'].values > self.limit_PR_WWA), self.limit_PR_WWA, sth['values'] )
            vals = sth[pri+'_values']#.values
            
            if method_synthesis == 'equality_datasets':
                sth[pri+'_mean'] = vals.mean( ('datasets', 'bootstrap') )
                sth[pri+'_median'] = vals.median( ('datasets', 'bootstrap') )
                for k in self.dico_q_confid:
                    sth[pri+'_'+k] = vals.quantile( q=self.dico_q_confid[k], dim=('datasets', 'bootstrap') ).drop('quantile')

            elif method_synthesis == 'equality_obs_models':
                # identify datasets
                obs_dat = [d for d in vals.datasets.values if d in self.obs_products]
                mod_dat = [d for d in vals.datasets.values if d not in self.obs_products]
                
                # calculate stats on observational datasets
                tmp = xr.Dataset()
                tmp['obs_mean'] = vals.sel(datasets=obs_dat).mean( ('datasets', 'bootstrap') )
                tmp['obs_median'] = vals.sel(datasets=obs_dat).median( ('datasets', 'bootstrap') )
                for k in self.dico_q_confid:
                    tmp['obs_'+k] = vals.sel(datasets=obs_dat).quantile( q=self.dico_q_confid[k], dim=('datasets', 'bootstrap') ).drop('quantile')
                    
                # calculate stats on model datasets
                tmp['mod_mean'] = vals.sel(datasets=mod_dat).mean( ('datasets', 'bootstrap') )
                tmp['mod_median'] = vals.sel(datasets=mod_dat).median( ('datasets', 'bootstrap') )
                for k in self.dico_q_confid:
                    tmp['mod_'+k] = vals.sel(datasets=mod_dat).quantile( q=self.dico_q_confid[k], dim=('datasets', 'bootstrap') ).drop('quantile')
                    
                # average
                sth[pri+'_mean'] = 0.5 * (tmp['obs_mean'] + tmp['mod_mean'])
                sth[pri+'_median'] = 0.5 * (tmp['obs_median'] + tmp['mod_median'])
                for k in self.dico_q_confid:
                    sth[pri+'_'+k] = 0.5 * (tmp['obs_'+k] + tmp['mod_'+k])

            else:
                # an alternative for calculating statistics: weights (log(PR) cf Philipp, but not cf internal doc with direct averaging), and factor_model_uncertainty for range not used in direct averaging
                raise Exception('Method not prepared')
            
        return sth

    def synthesis_global( self ):
        # preparation
        dict_values = {name_data:xr.Dataset() for name_data in self.datasets_postselection}
        for name_data in self.datasets_postselection:
            for var in self.evt_fits[name_data].PR[self.window].variables:
                dict_values[name_data]['PR_'+var] = self.evt_fits[name_data].PR[self.window][var]
            for var in self.evt_fits[name_data].I[self.window].variables:
                dict_values[name_data]['I_'+var] = self.evt_fits[name_data].I[self.window][var]
        self.limit_PR_WWA = self.evt_fits[self.ref].limit_PR_WWA
        
        # check before synthesis
        self.test_systematic_differences( values=[dict_values[name_data]['PR_median'].values for name_data in self.datasets_postselection] )
        
        # doing synthesis
        self.global_synthesis = self.func_synthesis( dict_values=dict_values )
    
    def eval_contribs_entities_GMT( self, dGMT_OSCAR, level_OSCAR='mean' ):
        self.contribs_GMT = {}
        self.level_OSCAR = level_OSCAR
        for i_data, name_data in enumerate(self.datasets_postselection):
            # preparing dataset
            self.contribs_GMT[name_data] = xr.Dataset()

            # preparing the sum over all entities for handling interaction terms
            if level_OSCAR == 'mean':
                gmt_all_entities = dGMT_OSCAR['dGMT_entities_mean'].sum('entity')
            elif level_OSCAR in [2.5, 50, 97.5]:
                raise Exception('would not be the sum of percentiles... need to recalculate the percentile on all.')
            else:
                raise Exception('not prepared: seen through percentiles low dependency, but for members, need to adapt the system of labels ("entity-cfg")')
            self.evt_fits[name_data].calc_params_bootstrap( predictor=self.data_to_do[name_data][1] - gmt_all_entities, label='no_entities' )
                                
            # looping on entities
            for i, entity in enumerate(dGMT_OSCAR.entity.values):
                if self.option_detailed_prints:
                    print( 'Preparing attribution based on data '+str(i_data+1)+'/'+str(len(self.datasets_postselection))+' entity '+str(i+1)+'/'+str(dGMT_OSCAR.entity.size)+' ', end='\r' )
                if level_OSCAR == 'mean':
                    gmt_minus = self.data_to_do[name_data][1] - dGMT_OSCAR['dGMT_entities_mean'].sel(entity=entity).drop('entity')
                    gmt_plus = self.data_to_do[name_data][1] - gmt_all_entities + dGMT_OSCAR['dGMT_entities_mean'].sel(entity=entity).drop('entity')
                elif level_OSCAR in [2.5, 50, 97.5]:
                    gmt_minus = self.data_to_do[name_data][1] - dGMT_OSCAR['dGMT_entities_percentiles'].sel(percentiles=level_OSCAR, entity=entity).drop('entity')
                    gmt_plus = self.data_to_do[name_data][1] - gmt_all_entities + dGMT_OSCAR['dGMT_entities_percentiles'].sel(percentiles=level_OSCAR, entity=entity).drop('entity')
                else:
                    raise Exception('not prepared: seen through percentiles low dependency, but for members, need to adapt the system of labels ("entity-cfg"), and still need to do the weighted average on that afterwards.')
                # calculating parameters for this entity, then predictor
                self.evt_fits[name_data].calc_params_bootstrap( predictor=gmt_minus, label='minus '+entity )
                self.evt_fits[name_data].calc_params_bootstrap( predictor=gmt_plus, label='plus '+entity )

            # deducing all probabilities & intensities
            self.evt_fits[name_data].full_calc_probas( evt_obs=self.evt_obs[name_data] )

            # calculating probability ratios of entities
            #vals = self.evt_fits[name_data].probabilities[self.window]['full'].sel(label='with_CC').drop('label') / self.evt_fits[name_data].probabilities[self.window]['full'].sel(label=dGMT_OSCAR.entity.values)
            # for some reasons, some indexing with entities raise an issue. Using isel instead of sel
            tmp = list(self.evt_fits[name_data].probabilities[self.window]['full'].label.values)
            ind_minus = [tmp.index('minus '+entity) for entity in dGMT_OSCAR.entity.values]
            ind_plus = [tmp.index('plus '+entity) for entity in dGMT_OSCAR.entity.values]
            p_full = self.evt_fits[name_data].probabilities[self.window]['full'].sel(label='with_CC').drop('label')
            p_fullminusentity = self.evt_fits[name_data].probabilities[self.window]['full'].isel(label=ind_minus).drop('label')
            p_onlyentity = self.evt_fits[name_data].probabilities[self.window]['full'].isel(label=ind_plus).drop('label')
            p_noentities = self.evt_fits[name_data].probabilities[self.window]['full'].sel(label='no_entities').drop('label')
            p_nat = self.evt_fits[name_data].probabilities[self.window]['full'].sel(label='without_CC').drop('label')
            i_full = self.evt_fits[name_data].intensities[self.window]['full'].sel(label='with_CC').drop('label')
            i_fullminusentity = self.evt_fits[name_data].intensities[self.window]['full'].isel(label=ind_minus).drop('label')
            i_onlyentity = self.evt_fits[name_data].intensities[self.window]['full'].isel(label=ind_plus).drop('label')
            i_noentities = self.evt_fits[name_data].intensities[self.window]['full'].sel(label='no_entities').drop('label')
            pr_vals = 0.5 * ( (p_full - p_fullminusentity) + (p_onlyentity - p_noentities) ) / p_nat
            pr_vals = pr_vals.rename( {'label':'entity'} )
            pr_vals.coords['entity'] = dGMT_OSCAR.entity.values
            self.contribs_GMT[name_data]['PR_values'] = pr_vals
            i_vals = 0.5 * ((i_full - i_fullminusentity) + (i_onlyentity - i_noentities))
            i_vals = i_vals.rename( {'label':'entity'} )
            i_vals.coords['entity'] = dGMT_OSCAR.entity.values
            self.contribs_GMT[name_data]['I_values'] = i_vals

            # calculating mean & median
            # blocking very high PR cf WWA approach: only for analysis, plots & interpretation, but not during calculations
            #pr_vals = xr.where( (pr_vals.values > self.limit_PR_WWA), self.limit_PR_WWA, pr_vals )
            self.contribs_GMT[name_data]['PR_mean'] = pr_vals.mean('bootstrap')
            self.contribs_GMT[name_data]['PR_median'] = pr_vals.median('bootstrap')
            self.contribs_GMT[name_data]['I_mean'] = i_vals.mean('bootstrap')
            self.contribs_GMT[name_data]['I_median'] = i_vals.median('bootstrap')

            # calculating confidence interval
            for q in self.evt_fits[name_data].dico_q_confid:
                self.contribs_GMT[name_data]['PR_'+q] = pr_vals.quantile(q=self.evt_fits[name_data].dico_q_confid[q], dim='bootstrap').drop('quantile')
                self.contribs_GMT[name_data]['I_' +q] = i_vals.quantile(q=self.evt_fits[name_data].dico_q_confid[q], dim='bootstrap').drop('quantile')
    
    
    def synthesis_entities_GMT( self ):
        self.contribs_GMT_synthesis = self.func_synthesis( dict_values=self.contribs_GMT )
        
    def eval_contribs_entities_CO2( self, emissions_FF, emissions_GCB ):
        self.contribs_CO2 = {}
        cum_emi_tot = (emissions_GCB['FF_CO2'] + emissions_GCB['LUC_CO2']).cumsum( 'year' )
        cum_emi = emissions_FF['emissions_CO2'].cumsum('year')
        ratio_cumemi = (cum_emi / cum_emi_tot).sel( year=self.evt_id.event_year )
        for i_data, name_data in enumerate(self.datasets_postselection):
            # preparing dataset
            self.contribs_CO2[name_data] = xr.Dataset()
            self.contribs_CO2[name_data]['PR_values'] = (self.evt_fits[name_data].PR[self.window]['values'] - 1) * ratio_cumemi
            self.contribs_CO2[name_data]['I_values'] = self.evt_fits[name_data].I[self.window]['values'] * ratio_cumemi
            
            # blocking very high PR cf WWA approach: only for analysis, plots & interpretation, but not during calculations
            #vals = self.PR_CO2[name_data]['values']
            #vals = xr.where( (vals.values > self.limit_PR_WWA), self.limit_PR_WWA, vals )
            self.contribs_CO2[name_data]['PR_mean'] = self.contribs_CO2[name_data]['PR_values'].mean('bootstrap')
            self.contribs_CO2[name_data]['PR_median'] = self.contribs_CO2[name_data]['PR_values'].median('bootstrap')
            self.contribs_CO2[name_data]['I_mean'] = self.contribs_CO2[name_data]['I_values'].mean('bootstrap')
            self.contribs_CO2[name_data]['I_median'] = self.contribs_CO2[name_data]['I_values'].median('bootstrap')

            # calculating confidence interval
            for q in self.evt_fits[name_data].dico_q_confid:
                self.contribs_CO2[name_data]['PR_'+q] = self.contribs_CO2[name_data]['PR_values'].quantile(q=self.evt_fits[name_data].dico_q_confid[q], dim='bootstrap').drop('quantile')
                self.contribs_CO2[name_data]['I_'+q] = self.contribs_CO2[name_data]['I_values'].quantile(q=self.evt_fits[name_data].dico_q_confid[q], dim='bootstrap').drop('quantile')
            
    def synthesis_entities_CO2( self ):
        self.contribs_CO2_synthesis = self.func_synthesis( dict_values=self.contribs_CO2 )
        
    def synthesis_entities( self, dGMT_OSCAR, emissions_FF, emissions_GCB, level_OSCAR='mean' ):
        self.eval_contribs_entities_GMT( dGMT_OSCAR, level_OSCAR=level_OSCAR )
        self.synthesis_entities_GMT()
        self.eval_contribs_entities_CO2( emissions_FF, emissions_GCB )
        self.synthesis_entities_CO2()
    #--------------------------------------------------------------

    
    #--------------------------------------------------------------
    # FILES, SAVE, LOAD
    def name_file_CO2(self, path_save, add_text=None):
        # checking path
        path_save = os.path.join(path_save, self.identifier_event)
        if not os.path.exists(path_save):os.makedirs(path_save)
            
        # name of file
        basis = 'synthesis-CO2_' + str(self.identifier_event)
        info_years = str(self.training_start_year) + '-' + str(self.training_end_year) + '-w' + self.evt_fits[self.ref].option_train_wo_event*'o' + 'evt'
        info_training = self.evt_fits[self.ref].weighted_NLL*'weighted' + 'NLL' + '-selected' + self.evt_fits[self.ref].select_BIC_or_NLL
        if self.n_ESMs_CMIP6 is None:
            info_sel = 'all-ESMs'
        else:
            info_sel = str(self.n_ESMs_CMIP6)+'-ESMs'
        if (add_text is None) or (add_text == ''):
            return os.path.join(path_save, basis + '_' + info_years + '_' + info_training + '_' + info_sel + '.nc')
        else:
            return os.path.join(path_save, basis + '_' + info_years + '_' + info_training + '_' + info_sel + '_' + add_text + '.nc')
        
    def name_file_GMT(self, path_save, add_text=None):
        # checking path
        path_save = os.path.join(path_save, self.identifier_event)
        if not os.path.exists(path_save):os.makedirs(path_save)
            
        # name of file
        basis = 'synthesis-GMT_' + str(self.identifier_event)
        info_years = str(self.training_start_year) + '-' + str(self.training_end_year) + '-w' + self.evt_fits[self.ref].option_train_wo_event*'o' + 'evt'
        info_training = self.evt_fits[self.ref].weighted_NLL*'weighted' + 'NLL' + '-selected' + self.evt_fits[self.ref].select_BIC_or_NLL
        if self.n_ESMs_CMIP6 is None:
            info_sel = 'all-ESMs'
        else:
            info_sel = str(self.n_ESMs_CMIP6)+'-ESMs'
        info_OSCAR = 'OSCAR-'+self.level_OSCAR
        if (add_text is None) or (add_text == ''):
            return os.path.join(path_save, basis + '_' + info_years + '_' + info_training + '_' + info_sel + '_' + info_OSCAR + '.nc')
        else:
            return os.path.join(path_save, basis + '_' + info_years + '_' + info_training + '_' + info_sel + '_' + info_OSCAR + '_' + add_text + '.nc')
        
    def save_synthesis(self, path_save, add_text=None):
        if self.WARNING_systematic_differences is not None:
            self.contribs_CO2_synthesis.attrs['WARNING_systematic_differences'] = self.WARNING_systematic_differences
            self.contribs_GMT_synthesis.attrs['WARNING_systematic_differences'] = self.WARNING_systematic_differences
        self.contribs_CO2_synthesis.to_netcdf( self.name_file_CO2(path_save=path_save, add_text=add_text), encoding={var: {"zlib": True} for var in self.contribs_CO2_synthesis.variables} )
        self.contribs_GMT_synthesis.to_netcdf( self.name_file_GMT(path_save=path_save, add_text=add_text), encoding={var: {"zlib": True} for var in self.contribs_GMT_synthesis.variables} )
        
    def test_load_all(self, path_save, level_OSCAR='mean', add_text=None ):
        self.level_OSCAR = level_OSCAR
        return os.path.isfile(self.name_file_CO2(path_save=path_save, add_text=add_text)) and os.path.isfile(self.name_file_GMT(path_save=path_save, add_text=add_text))
        
    def load_synthesis(self, path_save, level_OSCAR='mean', add_text=None):
        self.contribs_CO2_synthesis = xr.open_dataset(self.name_file_CO2(path_save=path_save, add_text=add_text))
        self.level_OSCAR = level_OSCAR
        self.contribs_GMT_synthesis = xr.open_dataset(self.name_file_GMT(path_save=path_save, add_text=add_text))
        if 'WARNING_systematic_differences' in self.contribs_CO2_synthesis.attrs:
            self.WARNING_systematic_differences = str(self.contribs_CO2_synthesis.attrs['WARNING_systematic_differences'])
        else:
            self.WARNING_systematic_differences = None
    #--------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------






#---------------------------------------------------------------------------------------------------------------------------
# CLASS PANORAMA = SUPER SYNTHESIS ON ALL DATASETS, ENTITIES and EVENTS
#---------------------------------------------------------------------------------------------------------------------------
class panorama:
    #--------------------------------------------------------------
    # BASIC FUNCTIONS
    def __init__(self, emissions_FF, dico_ISO2country, dico_country2reg,\
                 n_ESMs_CMIP6, reference, level_OSCAR, training_start_year_obs, training_end_year_CMIP6, option_train_wo_event, select_BIC_or_NLL, weighted_NLL ):
        self.emissions_FF = emissions_FF
        self.dico_ISO2country = dico_ISO2country
        self.dico_country2reg = dico_country2reg
        self.n_ESMs_CMIP6 = n_ESMs_CMIP6
        self.reference = reference
        self.level_OSCAR = level_OSCAR
        self.training_start_year_obs = training_start_year_obs
        self.training_end_year_CMIP6 = training_end_year_CMIP6
        self.option_train_wo_event = option_train_wo_event
        self.select_BIC_or_NLL = select_BIC_or_NLL
        self.weighted_NLL = weighted_NLL
    #--------------------------------------------------------------


    #--------------------------------------------------------------
    # gathering all values
    def gather_values( self, results ):
        self.results = results
        
        # prepare dataset
        self.contribs_all = xr.Dataset()
        keys = ['PR_median', 'PR_confid_bottom', 'PR_confid_upper', 'I_median', 'I_confid_bottom', 'I_confid_upper']
        
        # initialize dataset
        for k in keys:
            self.contribs_all['values_CO2_'+k] = xr.DataArray( np.nan, coords={'event':list(self.results.keys()), 'entity':self.emissions_FF.entity}, dims=('event','entity',))
            self.contribs_all['values_GMT_'+k] = xr.DataArray( np.nan, coords={'event':list(self.results.keys()), 'entity':self.emissions_FF.entity}, dims=('event','entity',))
            self.contribs_all['values_global_'+k] = xr.DataArray( np.nan, coords={'event':list(self.results.keys())}, dims=('event',))

        # fill in dataset
        for ind_evt in self.results.keys():
            for k in keys:
                self.contribs_all['values_CO2_'+k].loc[{'event':ind_evt}] = self.results[ind_evt][-1].contribs_CO2_synthesis[k]
                self.contribs_all['values_GMT_'+k].loc[{'event':ind_evt}] = self.results[ind_evt][-1].contribs_GMT_synthesis[k]
                self.contribs_all['values_global_'+k].loc[{'event':ind_evt}] = self.results[ind_evt][-1].global_synthesis[k]

    # calculate everything
    def eval_ISOs(self, eval_ISOs_from_results=True, emdat=None, indexes_events=None, geobounds=None, dict_geobounds=None, threshold_SequenceMatcher=None ):
        if eval_ISOs_from_results:
            self.dico_isos = {ind_evt:self.results[ind_evt][0].event_iso[0] for ind_evt in self.results.keys()}
        else:
            self.dico_isos = {}
            for ind_evt in indexes_events:
                evt_id = treat_event( evt=emdat.sel(index=ind_evt), geobounds=geobounds, dict_geobounds=dict_geobounds, threshold_SequenceMatcher=threshold_SequenceMatcher )
                self.dico_isos[ind_evt] = evt_id.eval_ISO_panorama()
    
    def calc(self):
        self.groups_global()
        self.groups_countries()
        self.groups_regions()
                
    # group into different countries: country of entity & country of event
    def groups_countries(self):
        # define dimensions country entities
        self.contribs_all['country_entity'] = xr.DataArray( self.emissions_FF['country_entity'].sel(entity=self.contribs_all['entity']).values, coords={'entity':self.contribs_all.entity}, dims=('entity',) )

        # define dimensions country event
        self.contribs_all['country_event'] = xr.DataArray( [self.dico_ISO2country[self.dico_isos[ind_evt]] for ind_evt in self.dico_isos], coords={'event':self.contribs_all.event}, dims=('event',) )

        # groupby by countries
        self.contribs_all_country_entity = self.contribs_all.groupby( self.contribs_all['country_entity'] ).sum()
        self.contribs_all_country_event = self.contribs_all.groupby( self.contribs_all['country_event'] ).mean()
                

    def groups_regions(self):
        # define dimensions region entities
        self.contribs_all['region_entity'] = xr.DataArray( [self.dico_country2reg[country] for country in self.contribs_all['country_entity'].values], coords={'entity':self.contribs_all.entity}, dims=('entity',) )

        # define dimensions country event
        self.contribs_all['region_event'] = xr.DataArray( [self.dico_country2reg[country] for country in self.contribs_all['country_event'].values], coords={'event':self.contribs_all.event}, dims=('event',) )

        # groupby by regions
        self.contribs_all_region_entity = self.contribs_all.groupby( self.contribs_all['region_entity'] ).sum()
        self.contribs_alll_region_event = self.contribs_all.groupby( self.contribs_all['region_event'] ).mean()

    def groups_global(self):
        # group on the globe
        self.contribs_all_entities = self.contribs_all.sum('entity')
        self.contribs_all_events = self.contribs_all.mean('event')
    #--------------------------------------------------------------
        
        
    #--------------------------------------------------------------
    # SAVE
    def name_file(self, add_text):
        basis = 'panorama'
        info_years = str(self.training_start_year_obs) + '-' + str(self.training_end_year_CMIP6) + '-w' + self.option_train_wo_event*'o' + 'evt'
        info_training = self.weighted_NLL*'weighted' + 'NLL' + '-selected' + self.select_BIC_or_NLL
        info_OSCAR = 'OSCAR-'+self.level_OSCAR
        if self.n_ESMs_CMIP6 is None:
            info_sel = 'all-ESMs'
        else:
            info_sel = str(self.n_ESMs_CMIP6)+'-ESMs'
        if (add_text is None) or (add_text == ''):
            return basis + '_' + info_years + '_' + info_training + '_' + info_sel + '_' + info_OSCAR + '.nc'
        else:
            return basis + '_' + info_years + '_' + info_training + '_' + info_sel + '_' + info_OSCAR + '_' + add_text + '.nc'
    
    def save_synthesis(self, path_save, add_text=None):
        name_file = self.name_file( add_text=add_text )
        self.contribs_all.to_netcdf( os.path.join(path_save, name_file), encoding={var: {"zlib": True} for var in self.contribs_all.variables} )
        
    def load_synthesis(self, path_save, add_text=None):
        name_file = self.name_file( add_text=add_text )
        self.contribs_all = xr.open_dataset(os.path.join(path_save, name_file))
        
    def test_load(self, path_save, add_text=None):
        name_file = self.name_file( add_text=add_text )
        return os.path.isfile( os.path.join(path_save, name_file) )
    #--------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------










#---------------------------------------------------------------------------------------------------------------------------
# TIERS OF CARBON MAJORS
#---------------------------------------------------------------------------------------------------------------------------    
def eval_tiers_emissionsCO2(emissions_FF, years_integration, fraction_tier1=0.5):
    emi_total = emissions_FF['emissions_CO2'].sel(year=years_integration).sum('year')
    sorted_entities = emi_total.argsort().values[::-1]
    cum_sorted_emi = emi_total.isel(entity=sorted_entities).cumsum('entity')
    cum_sorted_emi /= cum_sorted_emi.values[-1]
    entities_tier1 = cum_sorted_emi.entity[np.where( cum_sorted_emi.values <= fraction_tier1)[0]].values # highest contributors first, up to fraction_tier1 is excluded
    entities_tier2 = cum_sorted_emi.entity[np.where( cum_sorted_emi.values > fraction_tier1)[0]].values # highest contributors first, after fraction_tier1 is included
    return entities_tier1, entities_tier2


def eval_tiers_dGMT(dGMT_OSCAR, years_integration, fraction_tier1=0.5, level_OSCAR='mean'):
    if level_OSCAR == 'mean':
        final_dGMT = dGMT_OSCAR['dGMT_entities_mean'].sel(time=years_integration[-1])
    else:
        raise Exception('not prepared with this level_OSCAR: '+level_OSCAR)
    sorted_entities = final_dGMT.argsort().values[::-1]
    cum_sorted_dGMT = final_dGMT.isel(entity=sorted_entities).cumsum('entity')
    cum_sorted_dGMT /= cum_sorted_dGMT.values[-1]
    entities_tier1 = cum_sorted_dGMT.entity[np.where( cum_sorted_dGMT.values <= fraction_tier1)[0]].values # highest contributors first, up to fraction_tier1 is excluded
    entities_tier2 = cum_sorted_dGMT.entity[np.where( cum_sorted_dGMT.values > fraction_tier1)[0]].values # highest contributors first, after fraction_tier1 is included
    return entities_tier1, entities_tier2
#---------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------



