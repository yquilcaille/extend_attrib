import os
from itertools import product
from math import e

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import seaborn as sns  # # for colors
import xarray as xr
from sklearn.utils import resample
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize
from scipy.special import expi, gamma, gammainc, gammaincc, iv, zeta
from statsmodels.regression.linear_model import OLS

CB_color_cycle = sns.color_palette( 'colorblind', n_colors=10000 )
from fcts_support_basic import *


#---------------------------------------------------------------------------------------------------------------------------
# CLASS FOR FITTING ONE CONDITIONAL DISTRIBUTION: from MESMER-X
class distrib_cov:
    """Class for the fit of a distribution with evolutions of the parameters with covariant variables. This class fits evolution of these covariants, with different functions possible. For now, only linear ('linear') and sigmoid ('logistic', 'arctan', 'gudermannian', 'errorfct', 'generalizedlogistic', 'generalizedalgebraic').
    The fit of this distribution embeds a semi-analytical optimization of the first guess.

    model for the evolutions of the parameters of the GEV, here only with linear terms:
    * mu = loc0 + sum_i loc_i * cov_loc_i
    * sigma = scale0 + sum_i scale_i * cov_scale_i
    * shape = shape0 + sum_i shape_i * cov_shape_i
    
    If logistic terms are asked over the same parameter, they are assumed to be under the same exponential, for instance:
    * mu = loc0 + delta / (1 + exp(beta1 * cov_loc1 + beta2 * cov_loc2 - epsilon )))
    instead of:
    * mu = loc0 + delta1 / (1 + exp(beta1 * (cov_loc1-epsilon1))) + delta2 / (1 + exp(beta2 * (cov_loc2-epsilon2)))
    This approximation helps in providing a better first guess for these terms.
    

    Parameters
    ----------
    data : numpy array 1D
        vector of observations for fit of a distribution with covariates on parameters
        
    cov_preds : dict
        dictionary of covariate variables (3 keys, (cov_loc, cov_scale, cov_shape)), for dependencies of the parameters of the distribution. Each item of the dictionary must have the same shape as data.
        
    distr : str
        type of distribution to fit. For now, only GEV, gaussian and poisson are prepared.
        
    method_fit : str
        Type of algorithm used during the optimization, using the function 'minimize'. Default: 'Nelder-Mead'. HIGHLY RECOMMENDED, for its stability.
        
    xtol_req : float
        Accurracy of the fit. Interpreted differently depending on 'method_fit'. Default: 1e-3
        
    ftol_req : float
        
    maxiter : None or int
        Maximum number of iteration of the optimization. Default: 5000. Doubled if logistic asked.
        
    maxfev : None or int
        Maximum number of evaluation of the function during the optimization. Default: np.inf (important for relatively complex fits)
        
    fg_shape : float
        First guess for the semi-analytical optimization of the actual first guess of the distribution.
        
    prior_shape : float or None
        sets a gaussian prior for the shape parameter. prior_shape /2 = standard deviation of this gaussian prior. Default: 0, equivalent to None.
    
    option_silent : boolean
        just to avoid too many messages
        
    error_failedfit : boolean
        if True, will raise an issue if the fit failed
        
    Returns
    -------
    Once this class has been initialized, the go-to function is fit(). It returns the vector of solutions for the problem proposed.

    Notes
    -----
    - Assumptions:
    - Disclaimer:
    - TODO:
        - case of multiple logistic evolutions over the same parameter of the distribution: assumed to be under the same exponential, much easier to handle.
    """



    #--------------------------------------------------------------------------------
    # INITIALIZATION
    def __init__(self, data, cov_preds, distrib, prior_shape=0, data_add_test=None, cov_preds_add_test=None, min_proba_test=None, \
                 boundaries_coeffs={}, boundaries_params={}, weighted_NLL=True, \
                 option_silent=True, method_fit="Nelder-Mead", xtol_req=1e-3, ftol_req=1.e-6, maxiter=50000, maxfev=50000, error_failedfit=False):

        # data
        self.data = data
        self.add_test = (data_add_test is not None) and (cov_preds_add_test is not None) and (min_proba_test is not None)
        if (self.add_test == False) and ((data_add_test is not None) or (cov_preds_add_test is not None) or (min_proba_test is not None)):
            raise Exception("Only one of data_add_test, cov_preds_add_test & min_proba_test have been provided, not the three of them. Please correct.")
        self.data_add_test = data_add_test
        self.cov_preds_add_test = cov_preds_add_test
        self.min_proba_test = min_proba_test
        
        # covariates:
        self.possible_sigmoid_forms = ['logistic', 'arctan', 'gudermannian', 'errorfct', 'generalizedlogistic', 'generalizedalgebraic']
        tmp = {'params':[cov_type[len('cov_'):] for cov_type in cov_preds.keys() if cov_type!='transfo']}
        # preparing additional values for test only. useful if dont want to train on them (eg WWA w/o event), but still check that want additional values supported (eg event in support of distribution)
        if self.add_test:
            if list(cov_preds_add_test.keys()) != list(cov_preds.keys()):
                raise Exception("Must have same keys in cov_preds and cov_preds_add_test")
            tmp_test = {'params':[cov_type[len('cov_'):] for cov_type in cov_preds.keys() if cov_type!='transfo']}
        for typ in tmp['params']:
            cov_type = 'cov_'+typ
            
            # names of the covariates
            tmp[cov_type+'_names'] = [item[0] for item in cov_preds[cov_type]]
            
            # data for the covariates
            tmp[cov_type+'_data'] = [item[1] for item in cov_preds[cov_type]]
            if self.add_test:
                tmp_test[cov_type+'_data'] = [item[1] for item in cov_preds_add_test[cov_type]]
            
            # form of the fit for the covariates
            tmp[cov_type+'_form'] = [item[2] for item in cov_preds[cov_type]]
            # warning: assuming that if there are multiple logistic evolutions for the same parameter, they are under the same exponential.
            # coefficients associated
            tmp['coeffs_'+typ+'_names'] = [typ+'_0']
            for ii in np.arange(len(tmp[cov_type+'_names'])):
                if tmp[cov_type+'_form'][ii] == 'linear':
                    tmp['coeffs_'+typ+'_names'].append(typ+'_linear_' + tmp[cov_type+'_names'][ii])

                elif tmp[cov_type+'_form'][ii] in self.possible_sigmoid_forms:
                    form_sigmoid = tmp[cov_type+'_form'][ii]
                    if typ+'_'+form_sigmoid+'_asymptleft' not in tmp['coeffs_'+typ+'_names']:
                        tmp['coeffs_'+typ+'_names'].append(typ+'_'+form_sigmoid+'_asymptleft')
                        tmp['coeffs_'+typ+'_names'].append(typ+'_'+form_sigmoid+'_asymptright')
                        tmp['coeffs_'+typ+'_names'].append(typ+'_'+form_sigmoid+'_epsilon')
                        if form_sigmoid in ['generalizedlogistic', 'generalizedalgebraic']:
                            tmp['coeffs_'+typ+'_names'].append(typ+'_'+form_sigmoid+'_alpha')
                    tmp['coeffs_'+typ+'_names'].append(typ+'_'+form_sigmoid+'_lambda_' + tmp[cov_type+'_names'][ii])

                elif (type(tmp[cov_type+'_form'][ii]) == list) and (tmp[cov_type+'_form'][ii][0] == 'power'):
                    pwr = tmp[cov_type+'_form'][ii][1]
                    tmp['coeffs_'+typ+'_names'].append(typ+'_power'+str(pwr)+'_' + tmp[cov_type+'_names'][ii])
                    
                else:
                    raise Exception('Unknown form of fit detected.')
                    
        # check for a case not handled: on one parameter, several sigmoids asked, but from different kinds: that would be a mess for the evaluation of the coefficients on the parameters.
        for typ in tmp['params']:
            lst_forms = [form for form in tmp[cov_type+'_form'] if form in self.possible_sigmoid_forms]
            if len(set(lst_forms)) > 1:
                raise Exception('Please avoid asking for different types of sigmoid on 1 parameter.')
                
        # adding terms if transformation asked.
        if 'transfo' in cov_preds.keys():
            self.transfo = [True,cov_preds['transfo']]
            tmp['coeffs_transfo_names'] = ['transfo_asymptleft', 'transfo_asymptright']
            if self.transfo[1] in ['generalizedlogistic', 'generalizedalgebraic']:
                tmp['coeffs_transfo_names'].append( 'transfo_alpha' )
        else:
            self.transfo = [False,None]

        # saving that in a single variable
        self.cov = tmp
        if self.add_test:
            self.cov_test = tmp_test
                
        # full list of coefficients
        if distrib in ['gaussian']:
            tmp = self.cov['coeffs_loc_names'] + self.cov['coeffs_scale_names']
        elif distrib in ['GEV', 'GPD', 'skewnorm']:
            tmp = self.cov['coeffs_loc_names'] + self.cov['coeffs_scale_names'] + self.cov['coeffs_shape_names']
        elif distrib in ['poisson']:
            tmp = self.cov['coeffs_loc_names'] + self.cov['coeffs_mu_names']
        if self.transfo[0]:
            tmp = tmp + self.cov['coeffs_transfo_names']
        self.coeffs_names = tmp
        
        # arguments
        self.distrib = distrib
        if self.distrib in ['GEV', 'GPD', 'skewnorm']:
            self.fg_shape = {'GEV':-0.25, 'GPD':0.1, 'skewnorm':0}[self.distrib]
        self.method_fit = method_fit
        if self.distrib in ['gaussian']:
            self.boundaries_params = {'loc':[-np.inf, np.inf], 'scale':[0,np.inf]}
        elif self.distrib in ['GEV']:
            self.boundaries_params = {'loc':[-np.inf, np.inf], 'scale':[0,np.inf], 'shape':[-np.inf, 1/3]}# due to definition of standard deviation of a GEV
        elif self.distrib in ['GPD']:
            self.boundaries_params = {'loc':[-np.inf, np.inf], 'scale':[0,np.inf], 'shape':[-np.inf, 1/2]}# due to definition of standard deviation of a GPD
        elif self.distrib in ['skewnorm']:
            self.boundaries_params = {'loc':[-np.inf, np.inf], 'scale':[0,np.inf], 'shape':[-np.inf, np.inf]}
        elif self.distrib in ['poisson']:
            self.boundaries_params = {'loc':[-np.inf, np.inf], 'mu':[0,np.inf]}
        else:
            raise Exception("Distribution not prepared here")
        # integrating prescribed boundaries in parameters
        for param in self.boundaries_params:
            if param in boundaries_params:
                self.boundaries_params[param] = [ np.max( [boundaries_params[param][0], self.boundaries_params[param][0]] ), np.min( [boundaries_params[param][1], self.boundaries_params[param][1]] ) ]
        self.boundaries_coeffs = boundaries_coeffs # this one is more technical
        self.xtol_req = xtol_req
        self.ftol_req = ftol_req
        self.maxiter = maxiter # used to have np.inf, but sometimes, the fit doesnt work... and it is meant to.
        self.error_failedfit = error_failedfit
        self.maxfev = maxfev # used to have np.inf, but sometimes, the fit doesnt work... and it is meant to.
        if method_fit in ['dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact', 'COBYLA', 'SLSQP'] + ['CG', 'Newton-CG']:
            raise Exception('method for this fit not prepared, to avoid')
        else:
            self.name_xtol = {'BFGS':'xrtol', 'L-BFGS-B':'gtol', 'Nelder-Mead':'xatol', 'Powell':'xtol', 'TNC':'xtol', 'trust-constr':'xtol'}[ method_fit ]
            # 'CG' has nothing about it there
            # 'Newton-CG':'xtol'
            # TNC and trust-constr: better to use xtol?
            self.name_ftol = {'BFGS':'gtol', 'L-BFGS-B':'ftol', 'Nelder-Mead':'fatol', 'Powell':'ftol', 'TNC':'ftol', 'trust-constr':'gtol'}[ method_fit ]
            # 'Newton-CG' nothing about it there
            # 'CG':'gtol'
        
        # test on sizes of sample
        if self.data.ndim > 1:
            raise Exception('input data must be a vector') ## should do tests also for covariations.
        if (len(self.cov['cov_loc_data']) > 0) and (self.data.shape[0] != self.cov['cov_loc_data'][0].shape[0]):
            raise Exception('Sample of data and covariations have different sizes...')

        # prior for shape: normal distribution with sd=0 is not valid
        if type(prior_shape) in [tuple,list,np.ndarray]:
            # need to half the prior_shape to be consistent
            prior_shape[1] /= 2
            # return logpdf of a normal distribution
            self._prior_shape = ss.norm(loc=prior_shape[0], scale=prior_shape[1]).logpdf
            
        else:
            if np.isclose(prior_shape, 0):
                if option_silent == False:
                    print("setting prior_shape to None")
                prior_shape = None

            if prior_shape is None:
                # always return 0
                self._prior_shape = lambda x: 0.0
            else:
                # need to half the prior_shape to be consistent
                prior_shape /= 2
                # return logpdf of a normal distribution
                self._prior_shape = ss.norm(loc=0, scale=prior_shape).logpdf
                
        # Calculate weights to average NLL by density of drivers
        self.weighted_NLL = weighted_NLL
        self.weights_driver = self.eval_weights()
                
    def eval_weights(self):
        if self.weighted_NLL:
            tmp = {}
            for par in self.cov['params']:
                for i_driver, name_driver in enumerate( self.cov['cov_'+par+'_names'] ):
                    if name_driver not in tmp:
                        tmp[name_driver] = self.cov['cov_'+par+'_data'][i_driver]
            tmp = np.array( [tmp[ev] for ev in tmp] ).T
            if len(tmp) == 0:
                weights_driver = np.ones( self.data.shape )
            else:
                m1, m2 = np.nanmin(tmp, axis=0), np.nanmax(tmp, axis=0)
                gmt_hist, tmpb = np.histogramdd( sample=tmp, bins=[np.linspace((m1-0.05*(m2-m1))[i], (m2+0.05*(m2-m1))[i], int(tmp.shape[0])) for i in range(tmp.shape[1])] )
                gmt_bins_center = [0.5 * (tmpb[i][1:] + tmpb[i][:-1]) for i in range(tmp.shape[1])]
                interp = RegularGridInterpolator(points=gmt_bins_center, values=gmt_hist)
                weights_driver = 1 / interp( tmp )
            
        else:
            weights_driver = np.ones( self.data.shape )
        return weights_driver / np.sum( weights_driver )
    #--------------------------------------------------------------------------------

    
                
    #--------------------------------------------------------------------------------
    # FIRST GUESS
    @staticmethod
    def g(k, shape):
        return gamma(1 - k * shape)
    
    def GEV_mean(self, loc, scale, shape):
        # case: shape > 1
        out = np.inf * np.ones( loc.shape )
        # case: shape < 1
        ind = np.where(shape < 1)
        out[ind] = (loc + scale * (self.g(1,shape) - 1) / shape)[ind]
        # case: shape ~ 0
        ind = np.where(np.isclose(shape,0))
        out[ind] = (loc + scale * e)[ind]
        return out
    
    def GEV_var(self, loc, scale, shape):
        # case: shape > 1/2
        out = np.inf * np.ones( loc.shape )
        # case: shape < 1/2
        ind = np.where(shape < 1/2)
        out[ind] = (scale**2 * (self.g(2,shape) - self.g(1,shape)**2) / shape**2)[ind]
        # case : shape ~ 0
        ind = np.where(np.isclose(shape,0))
        out[ind] = (scale**2 * np.pi**2 / 6)[ind]
        return out

    def GEV_skew(self, loc, scale, shape):
        # case: shape > 1/3
        out = np.inf * np.ones( loc.shape )
        # case: shape < 1/3
        ind = np.where(shape < 1/3)
        out[ind] = ( np.sign(shape) * (self.g(3,shape) - 3 * self.g(2,shape) * self.g(1,shape) + 2 * self.g(1,shape) ** 3) / (self.g(2,shape) - self.g(1,shape)**2)**(3/2) )[ind]
        # case: shape ~ 0
        ind = np.where(np.isclose(shape,0))
        out[ind] = 12 * np.sqrt(6) * zeta(3) / np.pi**3
        return out

    @staticmethod
    def GPD_mean(loc, scale, shape):
        # case: shape >= 1
        out = np.inf * np.ones( loc.shape )
        # case: shape < 1
        ind = np.where(shape < 1)
        out[ind] = ( loc + scale / (1 - shape) )[ind]
        return out
    
    @staticmethod
    def GPD_var(loc, scale, shape):
        # case: shape >= 1/2
        out = np.inf * np.ones( loc.shape )
        # case: shape < 1/2
        ind = np.where(shape < 1/2)
        out[ind] = ( scale**2 / (1 - shape)**2 / (1 - 2*shape) )[ind]
        return out

    @staticmethod
    def GPD_skew(loc, scale, shape):
        # case: shape >= 1/3
        out = np.inf * np.ones( loc.shape )
        # case: shape < 1/3
        ind = np.where(shape < 1/3)
        out[ind] = ( 2 * (1 + shape) * np.sqrt(1 - 2*shape) / (1 - 3*shape) )[ind]
        return out
    
    def eval_1_fg0(self, arr):
        deltaloc, deltascale, shape = arr
        
        # checking these values, by taking 'self.fg_x0', with first estimate of parameters, including covariants of loc
        x0_test = np.copy( self.fg_x0 )
        x0_test[self.coeffs_names.index('loc_0')] += deltaloc # careful with that one
        x0_test[self.coeffs_names.index('scale_0')] += deltascale
        if 'shape_0' in self.coeffs_names:
            x0_test[self.coeffs_names.index('shape_0')] = shape
        args = self._parse_args(x0_test)
        test = self._test_coeffs(x0_test) * self._test_evol_params(self.data, args)
        
        if self.add_test:
            # additional data, not used for actual training, but still necessary to test the validity of the distribution
            add_args = self._parse_args(x0_test, option_test_data=True)
            test *= self._test_evol_params(self.data_add_test, add_args)
        
        if test:
            if self.distrib in ['GEV']:
                # calculating mean, median, skew that a GEV would have with these parameters
                err_mean = ( self.dd_mean - np.mean(self.GEV_mean(args[0], args[1], args[2])) )**2.
                err_var = (self.dd_var - np.mean(self.GEV_var(args[0], args[1], args[2])) )**2.
                err_skew = (self.dd_skew - np.mean(self.GEV_skew(args[0], args[1], args[2])) )**2.
                
            elif self.distrib in ['GPD']:
                # calculating mean, median, skew that a GEV would have with these parameters
                err_mean = ( self.dd_mean - np.mean(self.GPD_mean(args[0], args[1], args[2])) )**2.
                err_var = (self.dd_var - np.mean(self.GPD_var(args[0], args[1], args[2])) )**2.
                err_skew = (self.dd_skew - np.mean(self.GPD_skew(args[0], args[1], args[2])) )**2.
                
            else:
                raise Exception('This distribution has not been prepared.')

            out = err_mean + err_var + err_skew
        else:
            out = np.inf
        
        return out
    
    
    @staticmethod
    def find_m1_m2(dat, inds_sigm, data_sigm):
        # identifying the sigmoid term with the stronger variations on the sigmoid: used to identify the proper 'm1', which matters for the sigmoid transformation and loc_0
        tmp = []
        for ii in range(len(data_sigm)):
            ind = data_sigm[0].argsort() # here starts the moment where it is written as if a single sigmoid evolution
            ii = int(0.1*len(ind)) # taking average over 10% of higher and lower values to determine these two values
            tmp.append(np.abs( np.mean(dat[ind[:ii]]) - np.mean(dat[ind[-ii:]]) ))
        istrsigm = np.argmax(tmp)
        # identification of the overall evolution --> identification of min and max
        ind = data_sigm[istrsigm].argsort() # here starts the moment where it is written as if a single sigmoid evolution
        ii = int(0.1*len(ind)) # taking average over 10% of higher and lower values to determine these two values
        if np.mean(dat[ind[:ii]]) < np.mean(dat[ind[-ii:]]):# increasing sigmoid evolution
            m1 = np.min(dat)
            m2 = np.max(dat)
        else:# decreasing sigmoid evolution
            m1 = np.max(dat)
            m2 = np.min(dat)
        # increasing range of (m1,m2). The 2 following lines account for both signs of the derivative
        m1 += 0.01 * (m1-m2)
        m2 -= 0.01 * (m1-m2)
        return m1, m2
        
        
    def reglin_fg(self, typ_cov, data):
        # initiating
        self.tmp_sol[typ_cov] = np.zeros( len(self.cov['coeffs_'+typ_cov+'_names']) )
        
        #---------------------
        # linear contribution
        if len(self.cov['cov_'+typ_cov+'_names']) > 0:# checking that they are covariates
            inds_lin = [i for i,form in enumerate(self.cov['cov_'+typ_cov+'_form']) if form == 'linear']
        else:
            inds_lin = []
        if len(inds_lin) > 0:# checking that they are covariates on parameters with linear form
            data_lin = [self.cov['cov_'+typ_cov+'_data'][i] for i in inds_lin]
            ex = np.concatenate( [np.ones((len(data),1)), np.array(data_lin).T], axis=1 )
        else:
            ex = np.ones( len(data) ) # overkill, can simply remove mean.
        mod = OLS(exog=ex, endog=data)
        res = mod.fit()
        self.tmp_sol[typ_cov][0] = res.params[0]
        if len(inds_lin) > 0:# just filling in linear coefficients
            for i in np.arange( 1, len(res.params) ):
                cf = typ_cov+'_linear_'+self.cov['cov_'+typ_cov+'_names'][inds_lin[i-1]]
                self.tmp_sol[typ_cov][ self.cov['coeffs_'+typ_cov+'_names'].index(cf) ] = res.params[i]

        # detrending with linear evolution
        data_detrended = data - res.predict()
        #---------------------
        
        
        #---------------------
        # power contribution
        if len(self.cov['cov_'+typ_cov+'_names']) > 0:# checking that they are covariates
            inds_pow = [i for i,form in enumerate(self.cov['cov_'+typ_cov+'_form']) if (type(self.cov['cov_'+typ_cov+'_form'][i]) == list) and (self.cov['cov_'+typ_cov+'_form'][i][0] == 'power')]
        else:
            inds_pow = []
        if len(inds_pow) > 0:# checking that they are covariates on parameters with linear form
            data_pow = []
            for i in inds_pow:
                pwr = self.cov['cov_'+typ_cov+'_form'][i][1]
                data_pow.append( self.cov['cov_'+typ_cov+'_data'][i] ** pwr )
            ex = np.concatenate( [np.ones((len(data),1)), np.array(data_pow).T], axis=1 )
        else:
            ex = np.ones( len(data) ) # overkill, can simply remove mean.
        mod = OLS(exog=ex, endog=data_detrended)
        res = mod.fit()
        self.tmp_sol[typ_cov][0] += res.params[0]
        if len(inds_pow) > 0:# just filling in power coefficients
            for i in np.arange( 1, len(res.params) ):
                pwr = self.cov['cov_'+typ_cov+'_form'][inds_pow[i-1]][1]
                cf = typ_cov+'_power'+str(pwr)+'_'+self.cov['cov_'+typ_cov+'_names'][inds_pow[i-1]]
                self.tmp_sol[typ_cov][ self.cov['coeffs_'+typ_cov+'_names'].index(cf) ] = res.params[i]

        # detrending with power evolution
        data_detrended -= res.predict()
        #---------------------

        
        #---------------------
        # sigmoid contribution
        # principle: making a sigmoid transformation to evaluate parameters.
        if len(self.cov['cov_'+typ_cov+'_names']) > 0:# checking that they are covariates on this parameter
            inds_sigm = [i for i,form in enumerate(self.cov['cov_'+typ_cov+'_form']) if form in self.possible_sigmoid_forms]
        else:
            inds_sigm = []
        if len(inds_sigm) > 0:# checking that there are covariates on this parameter with lositic form
            # already made sure that only one sigmoid form is used on this parameter
            form_sigm = self.cov['cov_'+typ_cov+'_form'][inds_sigm[0]]
            # gathering data
            data_sigm = [self.cov['cov_'+typ_cov+'_data'][i] for i in inds_sigm]
            ex = np.concatenate( [np.ones((len(self.data),1)), np.array(data_sigm).T], axis=1 )

            # identifying boundaries of the sigmoid evolutions
            m1,m2 = self.find_m1_m2(data_detrended, inds_sigm, data_sigm)
            
            # sigmoid transformation
            if form_sigm in ['generalizedlogistic', 'generalizedalgebraic']:
                alpha = 1
                data_detrended_transf = sigmoid_transf(data=data_detrended, left=m1, right=m2, type_sigm=form_sigm, detect_NaN=False, alpha=alpha)
                self.tmp_sol[typ_cov][ self.cov['coeffs_'+typ_cov+'_names'].index(typ_cov+'_'+form_sigm+'_alpha') ] = alpha
            else:
                data_detrended_transf = sigmoid_transf(data=data_detrended, left=m1, right=m2, type_sigm=form_sigm, detect_NaN=False)
            # fitting
            mod = OLS(exog=ex, endog=data_detrended_transf)# assuming linear variations on these terms!
            res = mod.fit()
            # filling in sigmoid coefficients
            self.tmp_sol[typ_cov][ self.cov['coeffs_'+typ_cov+'_names'].index(typ_cov+'_'+form_sigm+'_asymptleft') ] = m1
            self.tmp_sol[typ_cov][ self.cov['coeffs_'+typ_cov+'_names'].index(typ_cov+'_'+form_sigm+'_asymptright') ] = m2
            for ii in range(len(data_sigm)):
                cv_nm = self.cov['cov_'+typ_cov+'_names'][inds_sigm[ii]]
                self.tmp_sol[typ_cov][ self.cov['coeffs_'+typ_cov+'_names'].index(typ_cov+'_'+form_sigm+'_lambda_'+cv_nm) ] = res.params[1+ii]
            self.tmp_sol[typ_cov][ self.cov['coeffs_'+typ_cov+'_names'].index(typ_cov+'_'+form_sigm+'_epsilon') ] = - res.params[0] # better physical interpretation, we are using anomalies in global variables

            # detrending with sigmoid evolution
            if form_sigm in ['generalizedlogistic', 'generalizedalgebraic']:
                data_detrended -= sigmoid_backtransf(res.predict(), m1, m2, type_sigm=form_sigm, detect_NaN=False, alpha=alpha)
            else:
                data_detrended -= sigmoid_backtransf(res.predict(), m1, m2, type_sigm=form_sigm, detect_NaN=False)
            
        else:
            pass # nothing to do
        #---------------------
        
        return data_detrended
        
        
    def find_fg(self):
        # Objective: optimizing loc_0, scale_0, shape_0, so that it matches the expected mean, variance and skewness. These values may be used as first guess for the real fit, especially the shape.
        # The use of the parameter "eval_fg0_best" leads to an effective doubling of this step. Even though this step is very fast, it could be avoided by saving both solutions in a self.thingy

        # INITIATING
        self.tmp_sol = {}
        
        # TRANFORMATION?
        if self.transfo[0]:
            # using all inputs on location to identify correct boundaries. More statistical sense with only linear terms?
            inds_sigm = np.arange(len(self.cov['cov_loc_form']))
            data_sigm = self.cov['cov_loc_data']

            # identifying boundaries of the sigmoid evolution 
            m1,m2 = self.find_m1_m2(self.data, inds_sigm, data_sigm)
            
            # transformation: new data that will be used
            if self.transfo[1] in ['generalizedlogistic', 'generalizedalgebraic']:
                # initiating alpha to 1
                alpha = 1
                data = sigmoid_transf(data=self.data, left=m1, right=m2, type_sigm=self.transfo[1], alpha=alpha )
                self.tmp_sol['transfo'] = [m1,m2,1] # transfo_asymptleft & right in coeffs_transfo_names
            else:
                data = sigmoid_transf(data=self.data, left=m1, right=m2, type_sigm=self.transfo[1])
                self.tmp_sol['transfo'] = [m1,m2] # transfo_asymptleft & right in coeffs_transfo_names
        else:
            data = self.data
        
        # LOCATION:
        data_det = self.reglin_fg( typ_cov='loc', data=data )
        
        # SCALE:
        # evaluating running standard deviation of scale on a nn-values window: there will be peaks at change in scenarios, but it will be smoothen out by the linear regressions
        # !!!!!!!!!!!! EDIT value of nn due to missing data in this exercice
        nn = 50 # 100 # with sma, seems to be better to have high values for proper evaluations of scale_0
        std_data_det = np.sqrt( np.max( [np.convolve( data_det**2 , np.ones(nn)/nn, mode='same') - np.convolve( data_det , np.ones(nn)/nn, mode='same')**2, np.zeros(data_det.shape)], axis=0 ) )
        if self.distrib in [ 'GEV', 'gaussian', 'GPD', 'skewnorm']:
            _ = self.reglin_fg( typ_cov='scale', data=std_data_det ) # coeffs are saved, useless to get the rest of the signal
        elif self.distrib in ['poisson']:
            _ = self.reglin_fg( typ_cov='mu', data=std_data_det ) # coeffs are saved, useless to get the rest of the signal
        else:
            raise Exception("Distribution not prepared here")
        
        # preparing optimization of loc0, scale0 and shape0: calculating mean, median, skew kurtosis of detrended data
        self.dd_mean = np.mean(data_det)
        self.dd_var = np.var(data_det)
        self.dd_skew = ss.skew(data_det)
        
        # finally creating the first guess
        if self.distrib in [ 'GEV' ]:
            
            # initialize first guess 'x0'. the full x0 will be used to test validity of the first guesses while optimizing.
            self.fg_x0 = list(self.tmp_sol['loc']) + list(self.tmp_sol['scale']) + [self.fg_shape] + list(np.zeros( len(self.cov['coeffs_shape_names'])-1 ))
            #self.fg_x0[self.coeffs_names.index('loc_0')] += self.dd_mean
            #self.fg_x0[self.coeffs_names.index('scale_0')] += np.sqrt( self.dd_var )
            if self.transfo[0]:
                self.fg_x0 = self.fg_x0 + list(self.tmp_sol['transfo'])

            # just avoiding zeros on sigmoid coefficients on shape parameter
            for param in self.cov['coeffs_shape_names']:
                for form in self.possible_sigmoid_forms:
                    if form+'_lambda' in param: # this parameter is a sigmoid lambda. Making a slow evolution
                        self.fg_x0[self.coeffs_names.index(param)] = 0.1
                    if form+'_asympt' in param: # this parameter is a sigmoid difference. Making a small difference, relatively to its parameter_0
                        if 'left' in param:
                            self.fg_x0[self.coeffs_names.index(param)] = 0.
                        else: #right
                            idcov = str.split( param, '_' )[0] + '_0'
                            self.fg_x0[self.coeffs_names.index(param)] = 0.01 * self.fg_x0[self.coeffs_names.index(idcov)]
            
            # preparing test of validity of starting domain
            x0_test = np.copy( self.fg_x0 )
            loc,scale,_ = self._parse_args(x0_test)
            
            # checking domain
            tmp = scale/(self.data-loc)
            bnds_c_limit = np.max(tmp[np.where(self.data-loc<0)]), np.min(tmp[np.where(self.data-loc>0)])
            if bnds_c_limit[1] < bnds_c_limit[0]:
                raise Exception("No possible solution for shape there ("+str(bnds_c_limit[0]) + " not < to " + str(bnds_c_limit[1])+"), please check.")
                
            # updating shape to confirm the validity of its domain, so that the fg_shape allows for a start within the support of the GEV
            correct_borders_shape = np.max( [-bnds_c_limit[1],self.boundaries_params['shape'][0]] ), np.min( [-bnds_c_limit[0],self.boundaries_params['shape'][1]] )
            self.safe_fg_shape = 0.5 * (correct_borders_shape[0] + correct_borders_shape[1]) # may need to take min or max +/- 0.1 for shape with covariants
            self.fg_x0[self.coeffs_names.index('shape_0')] = self.safe_fg_shape
            # 0.1 is meant to add some margin in the domain. The definition of the support assumes that the shape is <0, which is no issue, for fg_shape is usually <0
            
            # used to optimize using eval_1_fg0, but this operation has been discarded: defining the shape based on the domain returns better and more stable results.
            x0 = self.fg_x0
                            
        elif self.distrib in [ 'gaussian' ]:
            # special case: the relevant parameters from the gaussian can directly be deduced here:
            x0 = list(self.tmp_sol['loc']) + list(self.tmp_sol['scale'])
            x0[self.coeffs_names.index('loc_0')] += np.mean(data_det) # the location computed out of this step is NOT a proper location_0!
            if self.transfo[0]:
                x0 = x0 + list(self.tmp_sol['transfo'])
        
        elif self.distrib in [ 'poisson' ]:
            # special case: the relevant parameters from the gaussian can directly be deduced here:
            x0 = list(self.tmp_sol['loc']) + list(self.tmp_sol['mu'])
            x0[self.coeffs_names.index('loc_0')] -= np.var(data_det) # the location computed out of this step is NOT a proper location_0!
            x0[self.coeffs_names.index('mu_0')] += np.var(data_det) # the location computed out of this step is NOT a proper location_0!
            if self.transfo[0]:
                x0 = x0 + list(self.tmp_sol['transfo'])
            args = self._parse_args( x0 )
            if np.min(args[1]) < 0:
                x0[self.coeffs_names.index('mu_0')] -= np.min(args[1])*1.1
            # some points in the sample may be 0, and for a location too high, it would cause the first guess to start in a wrong domain
            counter_modif_loc0 = 0
            while np.isinf(self.loglike(x0)) and np.any(np.isclose(self.data,0)) and (counter_modif_loc0 < 10):
                x0[self.coeffs_names.index('loc_0')] -= 1
                counter_modif_loc0+= 1
        
        elif self.distrib in [ 'GPD' ]:
            # initialize first guess 'x0'. the full x0 will be used to test validity of the first guesses with an optimization.
            self.fg_x0 = list(self.tmp_sol['loc']) + list(self.tmp_sol['scale']) + [self.fg_shape] + list(np.zeros( len(self.cov['coeffs_shape_names'])-1 ))
            #self.fg_x0[self.coeffs_names.index('loc_0')] += self.dd_mean
            #self.fg_x0[self.coeffs_names.index('scale_0')] += np.sqrt( self.dd_var )
            if self.transfo[0]:
                self.fg_x0 = self.fg_x0 + list(self.tmp_sol['transfo'])

            # just avoiding zeros on sigmoid coefficients on shape parameter
            for param in self.cov['coeffs_shape_names']:
                for form in self.possible_sigmoid_forms:
                    if form+'_lambda' in param: # this parameter is a sigmoid lambda. Making a slow evolution
                        self.fg_x0[self.coeffs_names.index(param)] = 0.1
                    if form+'_asympt' in param: # this parameter is a sigmoid difference. Making a small difference, relatively to its parameter_0
                        if 'left' in param:
                            self.fg_x0[self.coeffs_names.index(param)] = 0.
                        else: #right
                            idcov = str.split( param, '_' )[0] + '_0'
                            self.fg_x0[self.coeffs_names.index(param)] = 0.01 * self.fg_x0[self.coeffs_names.index(idcov)]

            # calculating evolution of parameters to deduce a correct domain
            loc,scale,shape = self._parse_args(self.fg_x0)
            # lower bound
            lower = 1.1 * np.min( [0,np.min(self.data - loc)] )
            self.fg_x0[self.coeffs_names.index('loc_0')] += lower
            
            # upper bound: assuming shape is negative
            fg_x0_test = np.copy( self.fg_x0 )
            shape_min = np.max( -scale / (self.data - (loc+lower)) )
            fg_x0_test[self.coeffs_names.index('shape_0')] = np.min( [shape_min+0.05, self.boundaries_params['shape'][1]] ) + 0.05 # adding some margin just in case
            
            # used to optimize using eval_1_fg0, but this operation has been discarded: defining the shape based on the domain returns better and more stable results.
            if self.neg_loglike( fg_x0_test ) < self.neg_loglike( self.fg_x0 ):
                x0 = fg_x0_test
            else:
                x0 = self.fg_x0
            
            if False:
                # upper bound
                if np.any((self.data - (loc+lower - scale/shape)) > 0)  or  np.any(np.isclose(0, (self.data - (loc+lower - scale/shape)))):
                    shape_min = np.max( -scale / (self.data - (loc+lower)) )
                    self.fg_x0[self.coeffs_names.index('shape_0')] = np.min( [shape_min+0.05, self.boundaries_params['shape'][1]] ) + 0.05 # adding some margin just in case

                # initializations of the values that will be optimized
                xx0 = [0, 0, self.fg_x0[self.coeffs_names.index('shape_0')]]

                # looping over options for the first guess, breaking when have a correct one.
                # need to pass eventually transformed data to tests embedded within self.eval_1_fg_0
                # The first guess for this optimization is roughly done.
                m = minimize(
                                self.eval_1_fg0,
                                x0= xx0,
                                method=self.method_fit,
                                options={self.name_xtol: self.xtol_req, self.name_ftol: self.ftol_req},
                            )
                delta_loc0, delta_scale0, shape0 = m.x

                # allocating values to x0, already containing information on initial first value of loc_0 and initial coefficients on linear covariates of location
                x0 = np.copy(self.fg_x0)
                x0[self.coeffs_names.index('loc_0')] += delta_loc0 # the location computed out of this step is NOT a proper location_0!
                x0[self.coeffs_names.index('scale_0')] += delta_scale0 # better & safer results with += instead of =
                x0[self.coeffs_names.index('shape_0')] = shape0
        
        elif self.distrib in [ 'skewnorm' ]:
            # special case: the relevant parameters can directly be deduced here, and adding the shape is not an issue for a skew normal distribution
            x0 = list(self.tmp_sol['loc']) + list(self.tmp_sol['scale']) + [self.fg_shape] + list(np.zeros( len(self.cov['coeffs_shape_names'])-1 ))
            x0[self.coeffs_names.index('loc_0')] += np.mean(data_det) # the location computed out of this step is NOT a proper location_0!
            if self.transfo[0]:
                x0 = x0 + list(self.tmp_sol['transfo'])
        
        else:
            raise Exception("Distribution not prepared here")
        
        # checking whether succeeded, or need more work on first guess
        if np.isinf(self.loglike(x0)):
            raise Exception("Could not find an improved first guess")
        else:
            return x0
    #--------------------------------------------------------------------------------

    
    
    #--------------------------------------------------------------------------------
    # TEST COEFFICIENTS
    def _parse_args(self, args, option_test_data=False):
        # checking whether using test data or training data
        if option_test_data:
            data = self.data_add_test.shape
            cov_data = self.cov_test
        else:
            data = self.data.shape
            cov_data = self.cov
        
        # preparing
        coeffs = np.asarray(args).T
        
        # looping on every available parameter: loc, scale, shape
        tmp = {}
        for typ in self.cov['params']:
            
            # initializing the output
            tmp[typ] = coeffs[self.coeffs_names.index(typ+'_0')] * np.ones(data)
            
            # looping on every covariate of this parameter
            for ii in np.arange(len(self.cov['cov_'+typ+'_names'])):
                # checking the form of its equations and using the relevant coefficients
                name_cov = self.cov['cov_'+typ+'_names'][ii]
                
                if self.cov['cov_'+typ+'_form'][ii] == 'linear':
                    # THIS is the plain normal case.
                    tmp[typ] += coeffs[self.coeffs_names.index(typ+'_linear_'+name_cov)] * cov_data['cov_'+typ+'_data'][ii]

                elif self.cov['cov_'+typ+'_form'][ii] in self.possible_sigmoid_forms:
                    form_sigm = self.cov['cov_'+typ+'_form'][ii]
                    # all sigmoid terms are under the same exponential: they are dealt with the first time sigmoid is encountered on this parameter
                    if self.cov['cov_'+typ+'_form'].index( form_sigm ) == ii:# only the first one with sigmoid is returned here, checking if it is the one of the loop
                        # summing sigmoid terms
                        ind_sigm = np.where( np.array(self.cov['cov_'+typ+'_form']) == form_sigm )[0]
                        var = 0
                        for i in ind_sigm:
                            L = coeffs[self.coeffs_names.index(typ+'_'+form_sigm+'_lambda_'+self.cov['cov_'+typ+'_names'][i])]
                            var += L * cov_data['cov_'+typ+'_data'][i]
                        # dealing with the exponential
                        left = coeffs[self.coeffs_names.index(typ+'_'+form_sigm+'_asymptleft')]
                        right = coeffs[self.coeffs_names.index(typ+'_'+form_sigm+'_asymptright')]
                        eps = coeffs[self.coeffs_names.index(typ+'_'+form_sigm+'_epsilon')]
                        if np.isclose(left,right):
                            tmp[typ] += np.zeros(var.shape)# just for shape of tmp[typ]
                        else:
                            if form_sigm in ['generalizedlogistic', 'generalizedalgebraic']:
                                tmp[typ] += sigmoid_backtransf(data=var - eps, left=left, right=right, type_sigm=form_sigm, alpha=coeffs[self.coeffs_names.index(typ+'_'+form_sigm+'_alpha')])
                            else:
                                tmp[typ] += sigmoid_backtransf(data=var - eps, left=left, right=right, type_sigm=form_sigm)
                    else:# not the first sigmoid term on this parameter, already accounted for.
                        pass
                    
                elif (type(self.cov['cov_'+typ+'_form'][ii]) == list) and (self.cov['cov_'+typ+'_form'][ii][0] == 'power'):
                    pwr = self.cov['cov_'+typ+'_form'][ii][1]
                    tmp[typ] += coeffs[self.coeffs_names.index(typ+'_power'+str(pwr)+'_'+name_cov)] * cov_data['cov_'+typ+'_data'][ii] ** pwr
                    
                else:
                    raise Exception(self.cov['cov_'+typ+'_form'][ii]+" is not prepared!")
        
        if self.distrib in ['GEV']:
            # The parameter scale must be positive. CHOOSING to force it to zero, to avoid spurious fits
            pos_scale = np.max( [tmp['scale'], 1.e-9 * np.ones(tmp['scale'].shape)], axis=0 )
            # Warning, different sign convention than scipy: c = -shape!
            return (tmp['loc'], pos_scale, -tmp['shape'])

        elif self.distrib in ['gaussian']:
            # The parameter scale must be positive. CHOOSING to force it to zero, to avoid spurious fits
            pos_scale = np.max( [tmp['scale'], 1.e-9 * np.ones(tmp['scale'].shape)], axis=0 )
            return (tmp['loc'], pos_scale)

        elif self.distrib in ['poisson']:
            # the location must be integer values
            int_loc = np.array(np.round(tmp['loc'],0),dtype=int)
            # The parameter mu must be positive. CHOOSING to force it to zero, to avoid spurious fits
            # Besides, when loc and mu are close to zero, the probability of obtaining the value 0 is ~1-mu. Having mu=0 makes any value != 0 infinitely unlikely => setting a threshold on mu at 1.e-9, ie 1 / 1e9 years.
            pos_mu = np.max( [tmp['mu'], 1.e-9 * np.ones(tmp['mu'].shape)], axis=0 )
            return (int_loc, pos_mu)
        
        elif self.distrib in ['GPD']:
            # The parameter scale must be positive. CHOOSING to force it to zero, to avoid spurious fits
            pos_scale = np.max( [tmp['scale'], 1.e-9 * np.ones(tmp['scale'].shape)], axis=0 )
            return (tmp['loc'], pos_scale, tmp['shape'])
        
        elif self.distrib in ['skewnorm']:
            # The parameter scale must be positive. CHOOSING to force it to zero, to avoid spurious fits
            pos_scale = np.max( [tmp['scale'], 1.e-9 * np.ones(tmp['scale'].shape)], axis=0 )
            return (tmp['loc'], pos_scale, tmp['shape'])
        
    
    def _test_coeffs(self, args):
        # warning here, args are the coefficients
        
        # initialize test
        test = True
        
        # checking set boundaries on coeffs
        for coeff in self.boundaries_coeffs:
            low,top = self.boundaries_coeffs[coeff]
            cff = args[self.coeffs_names.index(coeff)]
            if np.any(cff < low) or np.any(top < cff) or np.any(np.isclose(cff , low)) or np.any(np.isclose(top , cff)):
                test = False # out of boundaries, strong signal to negative log likelyhood

        # checking the transformation
        if self.transfo[0]:
            left = args[self.coeffs_names.index('transfo_asymptleft')]
            right = args[self.coeffs_names.index('transfo_asymptright')]
            if left < right:
                if (np.min(self.data) < left)  or  (right < np.max(self.data)):
                    test = False
            else:
                if (np.max(self.data) > left)  or  (right > np.min(self.data)):
                    test = False
        
        # checking that the coefficient in the exponential of the sigmoid evolution is positive
        for param in self.coeffs_names:
            if '_lambda' in param: # on this parameter, there is a sigmoid evolution.
                pass
                #if args[self.coeffs_names.index(param)] < 0:
                #    test = False

        # has observed with sigmoid evolution that there may be compensation between coefficients, leading to a biased evolution of coefficients. This situation is rare for TXx, but happens often with SMA.
        for param in self.cov['params']:
            for form_sigm in self.possible_sigmoid_forms:
                if form_sigm in self.cov['cov_'+param+'_form']:
                    # checking for this parameter of the distribution if the fit of a sigmoid evolution leads to a drift in its param_0 and param_sigmoid_delta_covariate
                    # criteria: if the first guess provided to the optimization function has increased the constant term X times, it is unlikely and there may be a drift.
                    try:
                        if np.abs(args[self.coeffs_names.index(param+'_0')]) > 10 * np.abs(self.x0[self.coeffs_names.index(param+'_0')]):
                            test = False
                    except AttributeError:
                        # test on coefficients but to define first guess, not yet during fit. Thus, has not & cannot do this test yet.
                        pass
            
        return test

    
    def _test_evol_params(self, data_fit, p_args):
        # warning, args are here the evolution of parameters. And args[2] is 'c', not 'shape'
        # warning n2: because of the option 'transfo', data_fit is not necessarily self.data 
        
        if self.distrib in ['GEV', 'GPD']:
            loc, scale, c = p_args
            do_c = True
            
        elif self.distrib in ['skewnorm']:
            loc, scale, c = p_args
            do_c = False
            
        elif self.distrib in ['gaussian']:
            loc, scale = p_args
            do_c = False

        elif self.distrib in ['poisson']:
            loc, mu = p_args
            do_c = False
            
        else:
            raise Exception("Distribution not prepared here")

        # initialize test
        test = True
        
        if do_c:
            # test of the support of the distribution: is there any data out of the corresponding support?
            if self.distrib == 'GEV':
                # The support of the GEV is: [ loc - scale/shape ; +inf [ if shape>0  and ] -inf ; loc - scale/shape ] if shape<0
                # NB: checking the support with only '<' is not enough, not even '<='. Using 'isclose' avoids data points to be too close from boundaries, leading to unrealistic values in the ensuing processes.
                if np.any( scale + c * (loc - data_fit) <= 0 )  or  np.any(np.isclose( 0, scale + c * (loc - data_fit) )): # rewritten for simplicity as scale + c * (loc - data) > 0
                    test = False
            elif self.distrib == 'GPD':
                # The support of the GEV is: [ loc ; +inf [ if shape>=0  and ] loc ; loc - scale/shape ] if shape<0
                # NB: checking the support with only '<' is not enough, not even '<='. Using 'isclose' avoids data points to be too close from boundaries, leading to unrealistic values in the ensuing processes.
                if np.any( (data_fit - loc) <= 0 )  or  np.any(np.isclose( 0, (data_fit - loc) )):
                    test = False
                if test:
                    ind = np.where( c < 0 )
                    if np.any( (data_fit - loc + scale/c)[ind] >= 0 )  or  np.any(np.isclose( 0, (data_fit - loc + scale/c)[ind] )):
                        test = False

            # comparing to prescribed borders for shape
            if test:# if false, no need to test
                low,high = self.boundaries_params['shape']
                if self.distrib == 'GEV':
                    if np.any(-c < low) or np.any(high < -c) or np.any(np.isclose(-c , low)) or np.any(np.isclose(high , -c)):
                        test = False # out of boundaries, strong signal to negative log likelyhood
                elif self.distrib == 'GPD':
                    if np.any( c < low) or np.any(high <  c) or np.any(np.isclose( c , low)) or np.any(np.isclose(high ,  c)):
                        test = False # out of boundaries, strong signal to negative log likelyhood

        # scale should be strictly positive, or respect any other set boundaries
        if test:# if false, no need to test
            if self.distrib in ['GEV', 'gaussian', 'GPD', 'skewnorm']:
                low,high = self.boundaries_params['scale']
                if np.any(scale < low) or np.any(high < scale):# or np.any(np.isclose(scale , low)) or np.any(np.isclose(high , scale)): # trying without the isclose, cases with no evolutions, ie scale~=0
                    test = False # out of boundaries, strong signal to negative log likelyhood
            elif self.distrib in ['poisson']:
                low,high = self.boundaries_params['mu']
                if np.any(mu < low) or np.any(high < mu):# or np.any(np.isclose(scale , low)) or np.any(np.isclose(high , scale)): # trying without the isclose, cases with no evolutions, ie scale~=0
                    test = False # out of boundaries, strong signal to negative log likelyhood

        # location should respect set boundaries
        if test:# if false, no need to test
            low,high = self.boundaries_params['loc']
            if np.any(loc < low) or np.any(high < loc):# or np.any(np.isclose(loc , low)) or np.any(np.isclose(high , loc)):
                test = False # out of boundaries, strong signal to negative log likelyhood
            
        return test
    
    def _test_proba_value(self, data_test, p_args):
        # calculating CDF
        if self.distrib in ['GEV']:
            cdf = ss.genextreme.cdf(data_test, loc=p_args[0], scale=p_args[1], c=p_args[2] )

        elif self.distrib in ['gaussian']:
            cdf = ss.norm.cdf(data_test, loc=p_args[0], scale=p_args[1] )

        elif self.distrib in ['poisson']:
            cdf = ss.poisson.cdf(data_test, loc=p_args[0], mu=p_args[1] )

        elif self.distrib in ['GPD']:
            cdf = ss.genpareto.cdf(data_test, loc=p_args[0], scale=p_args[1], c=p_args[2] )

        elif self.distrib in ['skewnorm']:
            cdf = ss.skewnorm.cdf(data_test, loc=p_args[0], scale=p_args[1], a=p_args[2] )

        else:
            raise Exception('This distribution has not been prepared.')

        # tested values must have a minimum probability of occuring
        return np.all( 1 - cdf >= self.min_proba_test )
    #--------------------------------------------------------------------------------
    
    
    
    
    

    #--------------------------------------------------------------------------------
    # FIT
    def loglike(self, args):
        
        if self._test_coeffs(args):
            # transformation?
            if self.transfo[0]:
                m1 = args[self.coeffs_names.index('transfo_asymptleft')]
                m2 = args[self.coeffs_names.index('transfo_asymptright')]
                # new data that will be used
                if self.transfo[1] in ['generalizedlogistic', 'generalizedalgebraic']:
                    alpha = args[self.coeffs_names.index('transfo_alpha')]
                    data_fit = sigmoid_transf(self.data, left=m1, right=m2, type_sigm=self.transfo[1], alpha=alpha)
                else:
                    data_fit = sigmoid_transf(self.data, left=m1, right=m2, type_sigm=self.transfo[1])
            else:
                data_fit = self.data

            # log-likelihood
            p_args = self._parse_args(args)
            test = self._test_evol_params(data_fit, p_args)
            
            # checking whether needs to test additional data, without training on it
            if self.add_test:
                add_p_args = self._parse_args(args, option_test_data=True)
                # adding test on having the value being supported
                test *= self._test_evol_params(self.data_add_test, add_p_args)
                
                # adding test on probability of event
                test *= self._test_proba_value( data_test=self.data_add_test, p_args=add_p_args )
            
            # computing quality of the set of coefficients
            if test:
                # if here, then everything looks fine
                if self.optim == 'fcNLL':
                    # splitting data_fit into two sets of data using the given threshold
                    ind_ok_data, ind_stopped_data = self.stopping_rule( data_fit=data_fit, p_args=p_args )
                    self.tmp_info_fc = [data_fit, ind_stopped_data, p_args]

                else:
                    ind_ok_data = np.arange( len(data_fit) )
                
                # calculating LL
                if self.distrib in ['GEV']:
                    prior = self._prior_shape(p_args[2])
                    ll = ss.genextreme.logpdf(data_fit, loc=p_args[0], scale=p_args[1], c=p_args[2] )

                elif self.distrib in ['gaussian']:
                    prior = 0
                    ll = ss.norm.logpdf(data_fit, loc=p_args[0], scale=p_args[1] )

                elif self.distrib in ['poisson']:
                    prior = 0
                    ll = ss.poisson.logpmf(data_fit, loc=p_args[0], mu=p_args[1] )

                elif self.distrib in ['GPD']:
                    prior = self._prior_shape(p_args[2])
                    ll = ss.genpareto.logpdf(data_fit, loc=p_args[0], scale=p_args[1], c=p_args[2] )

                elif self.distrib in ['skewnorm']:
                    prior = self._prior_shape(p_args[2])
                    ll = ss.skewnorm.logpdf(data_fit, loc=p_args[0], scale=p_args[1], a=p_args[2] )
                    
                else:
                    raise Exception('This distribution has not been prepared.')
                # if ll+prior > 0, want to reduce output if quality transformation decrease
                # if ll+prior < 0, want to reduce further output if quality transformation decrease                
                return np.sum( (self.weights_driver * ll)[ind_ok_data] ) + prior

            else: # something wrong with the evolution of parameters
                return -np.inf
            
        else: # something wrong with the coefficients
            return -np.inf

    
    def neg_loglike(self, args):
        # negative log likelihood (for fit)
        # just in case used out of the optimization function, to evaluate the quality of the fit
        if type(args) == dict:
            args = [args[kk] for kk in args]
        
        return -self.loglike(args)
    
    
    def stopping_rule( self, data_fit, p_args ):
        # evaluating threshold over time
        if self.distrib in ['GEV']:
            thres = ss.genextreme.isf(q=1/self.threshold_stopping_rule, loc=p_args[0], scale=p_args[1], c=p_args[2])[self.ind_year_thres]

        elif self.distrib in ['gaussian']:
            thres = ss.norm.isf(q=1/self.threshold_stopping_rule, loc=p_args[0], scale=p_args[1])[self.ind_year_thres]

        elif self.distrib in ['poisson']:
            thres = ss.poisson.isf(q=1/self.threshold_stopping_rule, loc=p_args[0], mu=p_args[1])[self.ind_year_thres]

        elif self.distrib in ['GPD']:
            thres = ss.genpareto.logpdf(q=1/self.threshold_stopping_rule, loc=p_args[0], scale=p_args[1], c=p_args[2])[self.ind_year_thres]

        elif self.distrib in ['skewnorm']:
            thres = ss.skewnorm.isf(q=1/self.threshold_stopping_rule, loc=p_args[0], scale=p_args[1], a=p_args[2])[self.ind_year_thres]

        else:
            raise Exception('This distribution has not been prepared.')
        
        # identifying where exceedances occur
        if self.exclude_trigger:
            ind_stopped_data = np.where( data_fit > thres )[0]
        else:
            ind_stopped_data = np.where( data_fit >= thres )[0]
        ind_ok = [i for i in np.arange(len(data_fit)) if i not in ind_stopped_data]
        
        return ind_ok, ind_stopped_data

    
    def fullcond_thres(self):
        # getting terms calculated during NLL
        data_fit, ind_stopped_data, p_args = self.tmp_info_fc
        
        # calculating 2nd term for full conditional of the NLL
        if self.distrib in ['GEV']:
            fc1 = ss.genextreme.logcdf(x=data_fit, loc=p_args[0], scale=p_args[1], c=p_args[2] )
            fc2 = ss.genextreme.sf(x=data_fit, loc=p_args[0], scale=p_args[1], c=p_args[2] )

        elif self.distrib in ['gaussian']:
            fc1 = ss.norm.logcdf(x=data_fit, loc=p_args[0], scale=p_args[1] )
            fc2 = ss.norm.sf(x=data_fit, loc=p_args[0], scale=p_args[1] )

        elif self.distrib in ['poisson']:
            fc1 = ss.poisson.logcdf(x=data_fit, loc=p_args[0], mu=p_args[1] )
            fc2 = ss.poisson.sf(x=data_fit, loc=p_args[0], mu=p_args[1] )

        elif self.distrib in ['GPD']:
            fc1 = ss.genpareto.logcdf(x=data_fit, loc=p_args[0], scale=p_args[1], c=p_args[2] )
            fc2 = ss.genpareto.sf(x=data_fit, loc=p_args[0], scale=p_args[1], c=p_args[2] )

        elif self.distrib in ['skewnorm']:
            fc1 = ss.skewnorm.logcdf(x=data_fit, loc=p_args[0], scale=p_args[1], a=p_args[2] )
            fc2 = ss.skewnorm.sf(x=data_fit, loc=p_args[0], scale=p_args[1], a=p_args[2] )

        else:
            raise Exception('This distribution has not been prepared.')
        #return np.sum( (self.weights_driver * fc1)[ind_stopped_data] )
        return np.log( np.sum( (self.weights_driver * fc2)[ind_stopped_data] ) )
    
    
    def fullconditioning_neg_loglike(self, args):
        NLL = self.neg_loglike( args )
        FC = self.fullcond_thres()
        
        return NLL + FC
    
    def translate_m_sol(self, mx):
        sol = {}
        for cf in self.coeffs_names:
            sol[cf] = mx[self.coeffs_names.index(cf)]
        return sol
            
        
    def fit(self, optim='NLL', threshold_stopping_rule=None, exclude_trigger=True, ind_year_thres=None):
        self.optim = optim
        self.threshold_stopping_rule = threshold_stopping_rule
        self.exclude_trigger = exclude_trigger
        self.ind_year_thres = ind_year_thres
        
        # checking if actually need to fit...
        if np.any(np.isnan(self.data)):
            return self.translate_m_sol( np.nan*np.ones(len(self.coeffs_names)) )

        else:
            # Before fitting, need a good first guess, using 'find_fg'.
            self.x0 = self.find_fg()
        
            if self.optim == 'fcNLL':
                # fitting
                m = minimize(
                    self.fullconditioning_neg_loglike,
                    x0=self.x0,
                    method=self.method_fit,
                    options={"maxfev": self.maxfev, "maxiter": self.maxiter, self.name_xtol: self.xtol_req, self.name_ftol: self.ftol_req},
                )
                
            elif self.optim == 'NLL':
                # fitting
                m = minimize(
                    self.neg_loglike,
                    x0=self.x0,
                    method=self.method_fit,
                    options={"maxfev": self.maxfev, "maxiter": self.maxiter, self.name_xtol: self.xtol_req, self.name_ftol: self.ftol_req},
                )
                
            else:
                raise Exception('Unknown optimisation function')

            # checking if that one failed as well
            if self.error_failedfit and (m.success == False):
                raise Exception('The fast detrend provides with a valid first guess, but not good enough.')

            return self.translate_m_sol(m.x)
    #--------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------





#---------------------------------------------------------------------------------------------------------------------------
# FUNCTIONS USED ALONG CLASS (adapted from MESMER-X)

# not sure for this one...
def read_form_fit_distrib(form_fit_distrib):
    '''
    Takes form_fit_distrib and translates into an usable dictionnary
    '''
    
    descrip_fit = {}
    # separating transfo, loc, scale, shape
    ind_ = str.split( form_fit_distrib, '__' )
    for item in ind_:
        
        # separating key words
        ind_ii = str.split(item,'_')

        # preparing their part
        if ind_ii[0] not in descrip_fit:
            descrip_fit[ind_ii[0]] = {}

        # testing:
        if ind_ii[0] not in ['transfo', 'loc', 'scale', 'shape', 'mu']:
            raise Exception('Unrecognized keyword in form_fit_distrib: ' + ind_ii[0] )
        
        # separating contributions
        for jj,subit in enumerate(ind_ii[1:]):
            if (ind_ii[0] == 'transfo'):
                if len(ind_ii) > 2:
                    raise Exception('Too many key words for the transformation in form_fit_distrib')
                else:
                    descrip_fit['transfo'] = subit
            else:
                cov, form = str.split( subit, '-' )
                if form[:len('power')] == 'power':
                    pwr = eval( form[len('power'):] )
                    descrip_fit[ind_ii[0]][cov] = ['power',pwr]
                else:
                    descrip_fit[ind_ii[0]][cov] = form
                
    return descrip_fit


def eval_param_distrib( params, distrib, cov ):
    """ Transforms a set of parameters at gridpoints for an ensemble of covariants.

    Parameters
    ----------
    params : dict
        dictionary with parameters
    
    cov : dict of required inputs
    
    Returns
    -------
    params_all : dict

    Notes
    -----
    - Assumptions:
    - Disclaimer:
    - TODO:

    """
    
    # preparing distribution
    if distrib in ['gaussian']:# this transformation must be performed even if it is from normal to normal, for there are evolution of the parameters.
        params_list = ['loc', 'scale']

    elif distrib in ['GEV', 'GPD', 'skewnorm']:
        params_list = ['loc', 'scale', 'shape']

    elif distrib in ['poisson']:
        params_list = ['loc', 'mu']

    else:
        raise Exception('Distribution not prepared')
    
    # checking whether 1 sol or multiple to generate -- size of n axis
    params_used = {}
    nr_n = []
    for par in params:
        if type(params[par]) in [list, np.ndarray]:
            nr_n.append( len(params[par]) )
            params_used[par] = params[par]
        else:
            nr_n.append( 1 )
            params_used[par] = np.array( [params[par]] ) # small correction to facilitate the job
    if len(set(nr_n)) > 1:
        raise Exception("Multiple sizes in parameters?")
    else:
        nr_n = nr_n[0]
    
    # handling length of timeseries -- size of t axis
    nr_t = []
    for type_cov in params_list:
        for item in cov['cov_'+type_cov]:
            nr_t.append( len(item[1]) )
    if len(nr_t) == 0:
        nr_t = 1
    else:
        nr_t = np.max(nr_t)

    # for each one of the 3 types of parameters, constructing its evolutions.
    possible_sigmoid_forms = ['logistic', 'arctan', 'gudermannian', 'errorfct', 'generalizedlogistic', 'generalizedalgebraic']
    params_all = {}
    params_all = {}
    for type_cov in params_list:

        ## shape. Warning, different sign convention than scipy: c = -shape!
        # parameter to fill in: initialize with the constant term
        params_all[type_cov] = params_used[type_cov+'_0'][:, np.newaxis] * np.ones( (nr_n, nr_t) )

        # preparing loop on covariants
        list_covs = [item[0] for item in cov['cov_'+type_cov]]
        list_corresp_covs = [item[1][np.newaxis, :] for item in cov['cov_'+type_cov]]
        list_forms = [item[2] for item in cov['cov_'+type_cov]]

        # looping over covariants of fits. Will see here if some are missing.
        for ii in range( len(cov['cov_'+type_cov]) ):

            # adding this term to the parameters:
            if list_forms[ii] == 'linear':
                params_all[type_cov] += params_used[type_cov+'_linear_'+list_covs[ii]][:, np.newaxis] * list_corresp_covs[ii]

            elif list_forms[ii] in possible_sigmoid_forms:
                form_sigm = list_forms[ii]
                # all sigmoid terms are under the same exponential: they are dealt with the first time 'form_sigm' is encountered on this parameter
                if list_forms.index( form_sigm ) == ii:# only the first one with 'form_sigm' is returned here, checking if it is the one of the loop
                    # summing sigmoid terms
                    ind_sigm = np.where( np.array(list_forms) == form_sigm )[0]
                    var = np.array(0)
                    for i in ind_sigm:
                        L = params_used[type_cov+'_'+form_sigm+'_lambda_'+list_covs[i]][:, np.newaxis]
                        var = np.sum( [var, L * list_corresp_covs[i]], axis=0 )
                    # dealing with the exponential
                    left = params_used[type_cov+'_'+form_sigm+'_asymptleft'][:, np.newaxis]
                    right = params_used[type_cov+'_'+form_sigm+'_asymptright'][:, np.newaxis]
                    eps = params[type_cov+'_'+form_sigm+'_epsilon'][:, np.newaxis]
                    if form_sigm in ['generalizedlogistic', 'generalizedalgebraic']:
                        params_all[type_cov] += sigmoid_backtransf(data=var - eps, left=left, right=right, type_sigm=form_sigm, alpha=params_used[type_cov+'_'+form_sigm+'_alpha'][:, np.newaxis])
                    else:
                        params_all[type_cov] += sigmoid_backtransf(data=var - eps, left=left, right=right, type_sigm=form_sigm)
                else:# not the first 'form_sigm' term on this parameter, already accounted for.
                    pass

            elif (type(list_forms[ii]) == list) and (list_forms[ii][0] == 'power'):
                pwr = list_forms[ii][1]
                params_all[type_cov] += params_used[type_cov+'_power'+str(pwr)+'_'+list_covs[ii]][:, np.newaxis] * list_corresp_covs[ii] ** pwr
                # for pwr > 2, faster than "np.power()" and equivalent to "pow()"
                # for integer pwr, it is however faster to use "data * data * data * ..."
                # because integer powers would cause problems for covariants < 0, may directly limit to integer powers and use the former expression to accelerate computations on large datasets.

            else:
                raise Exception('Unknown evolution of coefficients')

        # corrections 
        if distrib in ['poisson']:
            if type_cov == 'loc':
                # the location must be integer values
                params_all[type_cov] = np.array(np.round(params_all[type_cov],0),dtype=int)
            elif type_cov == 'mu':
                # The parameter mu must be positive.
                # Besides, when loc and mu are close to zero, the probability of obtaining the value 0 is ~1-mu. Having mu=0 makes any value != 0 infinitely unlikely => setting a threshold on mu at 1.e-9, ie 1 / 1e9 years.
                params_all[type_cov] = np.max( [params_all[type_cov], 1.e-9 * np.ones(params_all[type_cov].shape)], axis=0 )

        elif distrib in ['gaussian', 'GEV', 'GPD', 'skewnorm']:
            if type_cov == 'scale':
                # The parameter scale must be positive.
                params_all[type_cov] = np.max( [params_all[type_cov], 1.e-9 * np.ones(params_all[type_cov].shape)], axis=0 )

    return params_all


def sigmoid_transf(data, left, right, type_sigm, alpha=None, detect_NaN=True):
    '''
    Transformation of a sigmoid into its term.
    For instance, for a logistic sigmoid, this function would perform a logistic transformation, the output being a logit.
    
    Available options for type_sigm:
        logistic, arctan, gudermannian, errorfct, generalizedlogistic, generalizedalgebraic
        - generalizedlogistic: needs alpha
        - generalizedalgebraic: needs alpha. not recommended, the back-transformation loses the sign.
        
    The last two take alpha as input, being a strictly positive real number.
    '''
    
    if np.isclose( left, right ):
        return np.zeros( data.shape )
    
    else:
        if type_sigm == 'logistic': # NB: hyperbolic tangent ~= logistic: 1/(1+np.exp(-x)) = 0.5*(1+np.tanh(x/2))
            yy = (data - left) / (right - left)
            #transf = -np.log( (1-yy) / yy )
            transf = np.log( (1-yy) / yy )

        elif type_sigm == 'arctan':
            yy = (data - (left+right)/2) / ((right - left)/2 * 2/np.pi)
            transf = np.tan( yy ) * 2 / np.pi

        elif type_sigm == 'gudermannian': # close to logistic
            yy = (data - (left+right)/2) / ((right - left)/2 * 4/np.pi)
            transf = np.arctanh( np.tan(yy) ) * 4/np.pi

        elif type_sigm == 'errorfct': # sharper transition than logistic
            yy = (data - (left+right)/2) / ((right - left)/2)
            transf = erfinv(yy)* 2 / np.sqrt(np.pi)

        elif type_sigm == 'generalizedlogistic': # if alpha==1, equivalent to logistic
            if alpha is None:
                raise Exception('A generalizedlogistic takes alpha as input')
            else:
                y1 = (data - left) / (right - left)
                y2 = pow(y1, 1/alpha)
                #transf = -np.log( (1-y2) / y2 ) + np.log(pow(2,1/alpha)-1)
                transf = np.log( (1-y2) / y2 ) + np.log(pow(2,1/alpha)-1)

        elif type_sigm == 'generalizedalgebraic':
            if alpha is None:
                raise Exception('A generalizedalgebraic takes alpha as input')
            else:
                yy = (data - (left+right)/2) / ((right - left)/2)
                transf = pow( pow(yy,alpha) / (1-pow(yy,alpha)), 1/alpha )

        else:
            raise Exception('Unknown type of sigmoid')

        if detect_NaN and np.any(np.isnan(transf)):
            raise Exception("NaN detected!")
        else:
            return transf


def sigmoid_backtransf(data, left, right, type_sigm, alpha=None, detect_NaN=True):
    '''
    Back-transformation of the central term into a sigmoig.
    For instance, for a logistic sigmoid, this function would take the logit and compute the logistic evolution.
    
    Available options for type_sigm:
        logistic, arctan, gudermannian, errorfct, generalizedlogistic, generalizedalgebraic
        - generalizedlogistic: needs alpha
        - generalizedalgebraic: needs alpha. not recommended, the back-transformation loses the sign.
        
    The last two take alpha as input, being a strictly positive real number.
    '''
    if type_sigm == 'logistic': # NB: hyperbolic tangent ~= logistic: 1/(1+np.exp(-x)) = 0.5*(1+np.tanh(x/2))
        #backtransf = left + (right - left) / (1 + np.exp(-data))
        backtransf = left + (right - left) / (1 + np.exp(data))

    elif type_sigm == 'arctan': # smoother transition than logistic
        backtransf = (left+right)/2 + (right-left)/2 * 2/np.pi * np.arctan(np.pi/2 * data)

    elif type_sigm == 'gudermannian': # close to logistic
        backtransf = (left+right)/2 + (right-left)/2 * 4/np.pi * np.arctan(np.tanh(np.pi/4 * data))

    elif type_sigm == 'errorfct': # sharper transition than logistic
        backtransf = (left+right)/2 + (right-left)/2 * erf(np.sqrt(np.pi)/2 * data)

    elif type_sigm == 'generalizedlogistic': # if alpha==1, equivalent to logistic
        if alpha is None:
            raise Exception('A generalizedlogistic takes alpha as input')
        else:
            #backtransf = left + (right-left) / (1+np.exp( -data + np.log(pow(2,1/alpha)-1) ))**alpha
            backtransf = left + (right-left) / (1+np.exp( data - np.log(pow(2,1/alpha)-1) ))**alpha

    elif type_sigm == 'generalizedalgebraic':
        if alpha is None:
            raise Exception('A generalizedalgebraic takes alpha as input')
        else:
            backtransf = (left+right)/2 + (right-left)/2 * data / pow((1+pow(np.abs(data),alpha)),1/alpha)

    else:
        raise Exception('Unknown type of sigmoid')

    # preparing case for left and right close, ie constant cases
    backtransf[...,np.isclose(left,right)] = 0.
        
    if detect_NaN and np.any(np.isnan(backtransf)):
        raise Exception("NaN detected!")
    else:
        return backtransf
    
    
def func_analytical_CRPS( obs, type_distrib, params ):
    """
    This function calculates the CRPS (Continuous Rank Probability Score) for an observed value and a GEV   OR   a gaussian.
     - The expression for the GEV is based on equation 9 of Friederichs et Thorarinsdottir, 2012 (DOI: 10.1002/env.2176).
     - The expression for the normal distribution is based on equation 8.55 of 'Statistical methods in the atmospherical sciences' of Wilks (p. 353)
     - POISSON? http://cran.nexr.com/web/packages/scoringRules/vignettes/crpsformulas.html
     - GPD eq11 & 12 of Friederichs et Thorarinsdottir, 2012 (DOI: 10.1002/env.2176).
     
    Assumptions:
     - The observation is provided without any distribution, hence implicitly following a Heaviside function. If this is not the case, the calculation of the CRPS must be properly rewritten.
     - Implied from the lower incomplete gamma function in the expression of eq.9 is that 1-shape_GEV > 0.
    """
    
    # shapes
    nr_n = [params[par].shape[0] for par in params]
    nr_t = [params[par].shape[1] for par in params]
    if (len(set(nr_n)) > 1) or (len(set(nr_t)) > 1):
        raise Exception('Shape of parameters are not identical')
    nr_n, nr_t = nr_n[0], nr_t[0]
    
    # check on time
    obs_used = obs[np.newaxis,:]
    
    # checking if constant parameters & time axis
    params_used = {}
    for par in params:
        if (nr_n,nr_t) == (1,1):
            params_used[par] = np.repeat( params[par], obs_used.shape[1], axis=1 )
        else:
            params_used[par] = params[par]
        
    
    # calculating CRPS
    if type_distrib == 'GEV':
        loc, scale, shape = params_used['loc'], params_used['scale'], params_used['shape']
        if shape.shape == (1,):
            shape = shape[0] * np.ones( obs_used.shape )
    
        # checking the shape in case where it is a GEV
        if np.any(shape > 1):
            raise Exception("Warning, the proper application of this function assumes that the shape of the GEV must be strictly below 1.")
        
        C = 0.5772156649
        
        # transforming
        obs_transf = (obs_used - loc) / scale
        crps_transf = np.nan * np.ones( obs_used.shape )
        cdf_obs_transf = ss.genextreme.cdf(x=obs_transf, loc=0, scale=1, c=-shape)
        ## handling case of very low probabilities causing infinite values in log
        log_cdf_obs_transf = -np.inf * np.ones( obs_transf.shape )
        log_cdf_obs_transf[np.where(np.isclose(cdf_obs_transf,0,atol=1e-300)==False)] = np.log(cdf_obs_transf[np.where(np.isclose(cdf_obs_transf,0,atol=1e-300)==False)])
                
        # shape close to 0
        i_s0 = np.where( np.isclose(shape, 0) )
        crps_transf[i_s0] =  -obs_transf[i_s0] - 2 * expi( log_cdf_obs_transf[i_s0] ) + (C-np.log(2))
        
        # preparation
        i_sN0 = np.where( np.isclose(shape, 0)==False )
        G = np.nan * np.ones( obs_used.shape )
        i_p = np.where( shape > 0 )
        i_c1 = np.where( obs_transf[i_p] <= -1/shape[i_p] )
        tmp = G[i_p]
        tmp[i_c1] = 0
        i_c2 = np.where( obs_transf[i_p] > -1/shape[i_p] )
        tmp[i_c2] = -cdf_obs_transf[i_p][i_c2] / shape[i_p][i_c2] + gammaincc( 1 - shape[i_p][i_c2], -log_cdf_obs_transf[i_p][i_c2] ) / shape[i_p][i_c2]
        G[i_p] = tmp
        i_n = np.where( shape < 0 )
        tmp = G[i_n]
        i_c1 = np.where( obs_transf[i_n] < -1/shape[i_n] )
        tmp[i_c1] = -cdf_obs_transf[i_n][i_c1] / shape[i_n][i_c1] + gammaincc( 1 - shape[i_n][i_c1], -log_cdf_obs_transf[i_n][i_c1] ) / shape[i_n][i_c1]
        i_c2 = np.where( obs_transf[i_n] >= -1/shape[i_n] )
        tmp[i_c2] = -1/shape[i_n][i_c2] + gamma(1 - shape[i_n][i_c2]) / shape[i_n][i_c2]
        G[i_n] = tmp
        # warning: 'gammaincc' is the *regularized* upper incomplete gamma function
        
        # rest of it
        crps_transf[i_sN0] = obs_transf[i_sN0] * (2 * cdf_obs_transf[i_sN0] - 1) - 2 * G[i_sN0] -  (1 - (2 - 2**shape[i_sN0]) * gamma(1-shape[i_sN0])) / shape[i_sN0]
        
        # back-transforming
        CRPS = scale * crps_transf

    elif type_distrib == 'gaussian':
        loc, scale = params_used['loc'], params_used['scale']        
        obs_transf = (obs_used-loc)/scale
        pdf_obs_transf = ss.norm.pdf(x=obs_transf, loc=0, scale=1)
        cdf_obs_transf = ss.norm.cdf(x=obs_transf, loc=0, scale=1)
        CRPS = scale * ( obs_transf * (2 * cdf_obs_transf - 1) + 2 * pdf_obs_transf - 1 / np.sqrt(np.pi) )

    elif type_distrib == 'poisson':
        loc, mu = params_used['loc'], params_used['mu']        
        obs_transf = obs_used - loc
        pdf_obs_transf = ss.poisson.pmf(x=np.floor(obs_transf), loc=0, mu=mu)
        cdf_obs_transf = ss.poisson.cdf(x=obs_transf, loc=0, mu=mu)
        CRPS = (obs_transf - mu) * (2 * cdf_obs_transf - 1) + 2 * mu * pdf_obs_transf - mu * np.exp( -2 * mu ) * (iv(v=0, z=2*mu) + iv(v=1, z=2*mu))

    elif type_distrib == 'GPD':
        raise Exception('some stuff about a threshold, not sure there...')
        
    # skewnorm not done
    else:
        raise Exception("Distribution not prepared")
        
    return CRPS
#---------------------------------------------------------------------------------------------------------------------------









#---------------------------------------------------------------------------------------------------------------------------
# CLASS FOR TRAINING ENSEMBLE OF DISTRIBUTIONS, SELECTING & BOOTSTRAPPING
class train_distribs:
    """
        to fill in
    """
    
    #--------------------------------------------------------------------------------
    # INITIALIZATION
    def __init__(self, data_gmt, data_obs, event_year, identifier_event, name_data, name_reference, training_start_year, training_end_year, potential_evolutions, potential_distributions, path_results, path_figures,\
                 xtol_req, weighted_NLL, select_BIC_or_NLL, option_train_wo_event, n_iterations_BS, option_detailed_prints=False ):
        # preparing list of windows
        self.list_windows = list( data_obs.keys() )
        self.training_start_year = training_start_year
        self.training_end_year = training_end_year
        self.option_detailed_prints = option_detailed_prints
        
        # noting that this dataset is the reference
        self.event_year = event_year
        self.identifier_event = identifier_event
        self.name_data = name_data
        self.name_reference = name_reference
        self.plot_trees = self.name_data == self.name_reference
        self.path_results = path_results
        self.path_figures = path_figures
        self.xtol_req = xtol_req
        self.weighted_NLL = weighted_NLL
        self.select_BIC_or_NLL = select_BIC_or_NLL
        self.option_train_wo_event = option_train_wo_event
        self.removed_years = [] + self.option_train_wo_event*[self.event_year]
        self.n_iterations_BS = n_iterations_BS
        
        # checking time axis
        if False:#np.any( [np.any(data_gmt.time.values != data_obs[window].time.values) for window in self.list_windows] ):
            raise Exception("Different time axis")
        else:
            # kept years
            yrs_in = []
            for yr in data_gmt.time.values:
                # higher than required training start year?
                if (self.training_start_year <= yr) and (yr <= self.training_end_year):
                    # year in common both in GMT and obs on all windows?
                    if np.all( [yr in data_obs[window].time for window in self.list_windows] ):
                        # no NaN on GMT and obs on all windows?
                        if np.isnan(data_gmt.sel(time=yr))==False  and  np.all( [np.isnan(data_obs[window].sel(time=yr))==False for window in self.list_windows] ):
                            yrs_in.append( yr )
            
            # selecting
            self.data_gmt = data_gmt.sel(time=yrs_in)
            self.data_obs = {window: data_obs[window].sel(time=yrs_in) for window in self.list_windows}
    
        # preparing (copy, to make sure that never edited because of reference/not reference issues)
        self.potential_evolutions = {k:np.copy(potential_evolutions[k]) for k in potential_evolutions}
        
        # possible distributions
        self.potential_distributions = potential_distributions
        self.distrib_params = {'gaussian': ['loc', 'scale'],  'GEV':['loc', 'scale', 'shape'],  'GPD':['loc', 'scale', 'shape'], 'skewnorm':['loc', 'scale', 'shape']}
        
        # testing whether probabilities are too low: event itself while training, and when calculating probabilities
        self.min_acceptable_proba = 1.e-9 # return period of 1 in a billion years deemed as more than long enough.
        self.limit_PR_WWA = 1.e4
    #--------------------------------------------------------------------------------
        
        
    #--------------------------------------------------------------------------------
    # functions used throughout the training
    def func_predictors( self, data ):
        if type( data ) == xr.core.dataarray.DataArray:
            if 'member' in data.coords:
                data = data.values.flatten()
            else:
                data = data.values
        
        # preparing shifted predictors
        data_m1 = np.hstack( [data[0], data[:-1]] )

        # possible evolutions for parameters
        prepared_evols = {
            'constant': [],\
            'linear':   [ ['GMT', data, 'linear'] ],\
            'power2':   [ ['GMT', data, ['power', 2]] ],\
            'poly2':    [ ['GMT', data, 'linear'],  ['GMT', data, ['power', 2]] ],\
            'power3':   [ ['GMT', data, ['power', 3]] ],\
            'poly3':    [ ['GMT', data, 'linear'],  ['GMT', data, ['power', 2]], ['GMT', data, ['power', 3]] ]
        }

        # checking for sigmoids
        if np.any( ['sigmoids' in self.potential_evolutions[pp] for pp in self.potential_evolutions] ):
            for sigm in ['logistic', 'arctan', 'gudermannian', 'errorfct', 'generalizedlogistic', 'generalizedalgebraic']:
                prepared_evols[sigm+'_m0'] = [ ['GMT', data, sigm] ]
                prepared_evols[sigm+'_m1'] = [ ['GMTm1', data_m1, sigm] ]
                prepared_evols[sigm+'_m0m1'] = [ ['GMT', data, sigm],  ['GMTm1', data_m1, sigm] ]
                
        return prepared_evols
    
    @staticmethod
    def func_boundaries_params( distrib ):
        # put in there constraints on parameters of conditional distribution that are not by default in "distrib_cov".
        if distrib == 'GEV':
            #return {'shape':[0,np.inf]}
            return {'shape':[-0.4,0.4]}
        else:
            return {}
    #--------------------------------------------------------------------------------

    
    
    #--------------------------------------------------------------------------------
    def fit_all( self ):
        # eventually removing additional years if required: used ONLY to check whether including or not the event would bias the modeling chain
        self.training_years = [yr for yr in self.data_gmt.time.values if yr not in self.removed_years]
        self.test_years = [self.event_year]# ALWAYS TESTING THIS YEAR NOW instead of: [yr for yr in self.data_gmt.time.values if yr in self.removed_years]
        
        # preparing list of predictors
        prepared_evols = self.func_predictors( data=self.data_gmt.sel(time=self.training_years) )
        prepared_evols_test = self.func_predictors( data=self.data_gmt.sel(time=self.test_years) )
        
        # preparing output for all fits:
        self.results_fits = {window:[] for window in self.list_windows}
        
        for distrib in self.potential_distributions:

            # generating list of combinations to iterate on
            params_list = self.distrib_params[ distrib ]
            if len(params_list) == 2:
                list_items = product(self.potential_evolutions[params_list[0]],\
                                     self.potential_evolutions[params_list[1]] )
            elif len(params_list) == 3:
                list_items = product(self.potential_evolutions[params_list[0]],\
                                     self.potential_evolutions[params_list[1]],\
                                     self.potential_evolutions[params_list[2]] )

            # looping on all potential combinations
            counter = 0
            for item in list_items: # product() items have no len, thus cant enumerate to have position

                # generating predictor for this fit
                tmp_preds, tmp_preds_test = {}, {}
                for i in range( len(params_list) ):
                    tmp_preds[ 'cov_'+params_list[i] ] = prepared_evols[ item[i] ]
                    tmp_preds_test[ 'cov_'+params_list[i] ] = prepared_evols_test[ item[i] ]

                # looping on windows
                for window in self.list_windows:
                    if 'member' in self.data_obs[window].coords:
                        data_fit = self.data_obs[window].sel(time=self.training_years).values.flatten()
                        data_fit_test = self.data_obs[window].sel(time=self.test_years).values.flatten()
                    else:
                        data_fit = self.data_obs[window].sel(time=self.training_years).values
                        data_fit_test = self.data_obs[window].sel(time=self.test_years).values
                    # fit
                    tmp_fit = distrib_cov(data=data_fit, cov_preds=tmp_preds, distrib=distrib, xtol_req=self.xtol_req, weighted_NLL=self.weighted_NLL, boundaries_params=self.func_boundaries_params(distrib),\
                                          data_add_test=data_fit_test, cov_preds_add_test=tmp_preds_test, min_proba_test=self.min_acceptable_proba )
                    sol = tmp_fit.fit()

                    # check success
                    if np.any( np.isnan(list(sol.values())) ):
                        raise Exception("Error with fit!")

                    # calculating variations of parameters
                    pars = eval_param_distrib( params=sol, distrib=distrib, cov=tmp_preds )

                    # scores for quality of fit
                    NLL = tmp_fit.neg_loglike( sol ) # already averaged over number of points
                    #CRPS = func_analytical_CRPS( obs=self.data_obs[window].sel(time=self.training_years).values, type_distrib=distrib, params=pars ).sum()
                    BIC = len(sol) * np.log(len(data_fit)) / len(data_fit) - 2 * (-NLL) # average BIC over number of points

                    # saving
                    self.results_fits[window].append( ['_'.join([distrib]+list(item)), sol, NLL, BIC] ) # CRPS
                    if window == self.list_windows[0]:
                        counter += 1
            if self.option_detailed_prints:
                print( 'Tried fits on '+self.name_data+': ' + distrib + ', ' + str(counter) + ' combinations done over ' + str(len(self.list_windows)) + ' windows', end='\n' )

    def select_best_fit( self ):
        # preparing output
        self.best_fit = { window:{} for window in self.list_windows }
        
        # looping on windows
        for window in self.list_windows:
            # selecting best configuration: based on the Bayesian Information Criteria
            if self.select_BIC_or_NLL == 'BIC':
                best_fit = self.results_fits[window][ np.argmin( [fit[3] for fit in self.results_fits[window]] ) ]
            elif self.select_BIC_or_NLL == 'NLL':
                best_fit = self.results_fits[window][ np.argmin( [fit[2] for fit in self.results_fits[window]] ) ]

            # reshaping for easier usages
            distrib = best_fit[0].split('_')[0]
            item_fit = best_fit[0].split('_')[1:]
            self.best_fit[window] = { 'distrib':distrib, 'items_fit':item_fit, 'parameters':best_fit[1], 'NLL':best_fit[2], 'BIC':best_fit[3] }#, 'CRPS':best_fit[4]

            # checking if any potential issue
            if np.all( [k=='constant' for k in self.best_fit[window]['items_fit']] ):
                #raise Exception("Best fit seems to be with constant parameters: double-check there.")
                if self.option_detailed_prints:
                    print("WARNING: Best fit seems to be with constant parameters: double-check there.")
    #--------------------------------------------------------------------------------

    
    #--------------------------------------------------------------------------------
    # BOOTSTRAPPING
    def bootstrap_best_fit( self ):
        # initialize
        self.sol_bootstrap = {window:{} for window in self.list_windows}
        
        # preparing
        if 'member' in self.data_gmt.coords:
            gmt_fit = self.data_gmt.sel(time=self.training_years).values.flatten()
        else:
            gmt_fit = self.data_gmt.sel(time=self.training_years).values
            
        # looping on windows
        for i_iter in range( self.n_iterations_BS ):
            if self.option_detailed_prints:
                print( 'bootstrap: '+str(np.round(100 * (i_iter+1) / self.n_iterations_BS, 1))+'%', end='\r' )

            # resampling the indexes of these points
            ind_train = resample( np.arange(len(gmt_fit)), replace=True )

            for window in self.list_windows:
                
                # preparing
                if 'member' in self.data_obs[window].coords:
                    data_fit = self.data_obs[window].sel(time=self.training_years).values.flatten()
                    data_fit_test = self.data_obs[window].sel(time=self.test_years).values.flatten()
                else:
                    data_fit = self.data_obs[window].sel(time=self.training_years).values
                    data_fit_test = self.data_obs[window].sel(time=self.test_years).values
                    
                # selecting the corresponding GMT & OBS
                boot_gmt = gmt_fit[ind_train]
                boot_obs = data_fit[ind_train]
                
                # preparing list of predictors
                prepared_evols = self.func_predictors( data=boot_gmt )
                prepared_evols_test = self.func_predictors( data=self.data_gmt.sel(time=self.test_years) ) # unchanged, just for test
                params_list = self.distrib_params[ self.best_fit[window]['distrib'] ]

                # generating predictor for this fit
                tmp_preds, tmp_preds_test = {}, {}
                for i in range( len(params_list) ):
                    tmp_preds[ 'cov_'+params_list[i] ] = prepared_evols[ self.best_fit[window]['items_fit'][i] ]
                    tmp_preds_test[ 'cov_'+params_list[i] ] = prepared_evols_test[ self.best_fit[window]['items_fit'][i] ]

                # fitting
                tmp_fit = distrib_cov(data=boot_obs, cov_preds=tmp_preds, distrib=self.best_fit[window]['distrib'], xtol_req=self.xtol_req, weighted_NLL=self.weighted_NLL,\
                                      boundaries_params=self.func_boundaries_params(self.best_fit[window]['distrib']), data_add_test=data_fit_test, cov_preds_add_test=tmp_preds_test, min_proba_test=self.min_acceptable_proba )
                sol = tmp_fit.fit()

                # archiving
                for param in sol:
                    # creating key the first time
                    if param not in self.sol_bootstrap[window]:
                        self.sol_bootstrap[window][param] = []
                    self.sol_bootstrap[window][param].append( sol[param] )

        # reshaping for easier uses
        for window in self.list_windows:
            for param in self.sol_bootstrap[window]:
                self.sol_bootstrap[window][param] = np.array( self.sol_bootstrap[window][param] )

        # controling quality of boostrap
        if False:
            # not trivial: complex fits (e.g. combining poly2 & logistic evolutions) tend to have more complex evolutions in the parameters
            self.detect_error_boostrap()
        if self.option_detailed_prints:
            print( 'Finished bootstraping of '+self.name_data )

    
    def detect_error_boostrap( self, threshold_error_bootstrap=1 ):
        # looping on windows
        for window in self.list_windows:
            for param in self.sol_bootstrap[window]:
                test_mm = np.mean( self.sol_bootstrap[window][param] )
                test_ss = np.std( self.sol_bootstrap[window][param] )

                if np.abs(self.best_fit[window]['parameters'][param] - test_mm) > threshold_error_bootstrap * test_ss:
                    raise Exception('Error during bootstrapping?')
    
    
    def calc_params_bootstrap( self, predictor, label ):
        # evolutions possible
        if 'member' in predictor.coords:
            prepared_evols = self.func_predictors( data=predictor.mean('member') )
        else:
            prepared_evols = self.func_predictors( data=predictor )
        
        # looping on windows
        xr_params_bootstrap = {window:xr.Dataset() for window in self.list_windows}
        for window in self.list_windows:
            # generating predictor for this fit
            params_list = self.distrib_params[ self.best_fit[window]['distrib'] ]
            tmp_preds = {}
            for i in range( len(params_list) ):
                tmp_preds[ 'cov_'+params_list[i] ] = prepared_evols[ self.best_fit[window]['items_fit'][i] ]

            # calculating variations of parameters
            out = eval_param_distrib( params=self.sol_bootstrap[window], distrib=self.best_fit[window]['distrib'], cov=tmp_preds )
            
            # dealing with case where all parameters are constant
            if np.all([it=='constant' for it in self.best_fit[window]['items_fit']]):
                out = {param: np.repeat(out[param],predictor.time.size,axis=1) for param in out}
            
            # finishing them
            for param in out:
                xr_params_bootstrap[window][param] = xr.DataArray( out[param], coords={'bootstrap':np.arange(self.n_iterations_BS), 'time':predictor.time}, dims=['bootstrap', 'time'] )
                
        # if first time called, will create a new variable, self.parameters, storing all tested solutions, with given label
        # if not first time, will append that to former one, with given label
        if hasattr(self, 'parameters') == False:
            self.parameters = {window: xr_params_bootstrap[window].expand_dims({'label':[label]}) for window in self.list_windows}
            
        else:
            for window in self.list_windows:
                self.parameters[window] = xr.concat( [self.parameters[window], xr_params_bootstrap[window].expand_dims({'label':[label]})], dim='label')

                
    def simple_plot_bootstrap( self, predictor ):
        fontsize = 12
        fig_bs = plt.figure( figsize=(20,10) )
        for i_window, window in enumerate(self.list_windows):
            params = list( self.sol_bootstrap[window].keys() )
            for i_param, param in enumerate(params):
                ax = plt.subplot( len(self.list_windows), len(params), len(params)*i_window + i_param+1 )
                # distrib of bootstrapped
                _ = plt.hist( self.sol_bootstrap[window][param], density=True, bins=50, label='bootstrapped distribution' )
                # best fit
                yl = ax.get_ylim()
                plt.vlines( x=[self.best_fit[window]['parameters'][param]], ymin=yl[0], ymax=yl[1], color='black', lw=3, ls=':', label='best fit' )

                # polishing
                plt.grid()
                plt.legend(loc=0,prop={'size':fontsize})
                plt.xlabel( 'Parameter #'+str(i_param+1), size=fontsize)
                plt.xticks(size=fontsize)
                _ = plt.xticks( size=fontsize )
                ax.tick_params(axis='y',label1On=False)
                if i_param == 0:
                    label = self.write_fit(self.best_fit[window], window)
                    plt.ylabel( label, size=fontsize)
                pos = ax.get_position()
                ax.set_position([pos.x0, pos.y0 , pos.width, 0.9*pos.height])
        return fig_bs
    #--------------------------------------------------------------------------------


    
    #--------------------------------------------------------------------------------
    def learn_from_ref( self, ref ):
        # test regarding simplification
        if len(self.list_windows) > 1:
            raise Exception("Would need to have a potential_distributions and potential_evolutions for each window... at the moment, written for just one to simplify the code.")
            
        # restraining the set of configurations to the one of the best fit
        for window in self.list_windows:
            # taking as only distribution the one from best fit
            self.potential_distributions = [ ref.best_fit[window]['distrib'] ]
            
            # taking as only evolutions the ones from best fit
            self.potential_evolutions = {}
            for i, param in enumerate( self.distrib_params[ ref.best_fit[window]['distrib'] ] ):
                self.potential_evolutions[param] = [ ref.best_fit[window]['items_fit'][i] ]
                
        # setting also probability level: return period of this dataset set as return period of reference
        # useful for CMIP6 ESMs: despite having a potentially realistic statistical distribution, the realisation by the ESM at time of the event is likely different from the value observed.
        # taking median here, instead of full distribution. Oberved to increase ranges for low probability events, though not changing much center of distribution of probabilities.
        self.event_probability = {window: ref.probabilities[window]['median'].sel(label='with_CC').drop('label') for window in self.list_windows}
        if self.event_year != ref.event_year:
            raise Exception("Supposed to have same year for definition of events!")


    def eval_event_level( self ):
        if self.name_data == self.name_reference:
            # direct solution for reference: this is the observed event for the reference dataset
            self.event_level = { window: self.evt_obs[window].sel(time=self.event_year).values for window in self.list_windows }
            
        else:
            # indirect solution for non-reference (e.g. CMIP6): using observed median return period of the event to evaluate median level of the event
            self.event_level = {}
            
            for window in self.list_windows:
                if self.best_fit[window]['distrib'] == 'GEV':
                    self.event_level[window] = ss.genextreme.ppf(q=1 - self.event_probability[window],\
                                                                 loc=self.parameters[window]['loc'].sel(label='with_CC', time=self.event_year),\
                                                                 scale=self.parameters[window]['scale'].sel(label='with_CC', time=self.event_year),\
                                                                 c=-self.parameters[window]['shape'].sel(label='with_CC', time=self.event_year) )

                elif self.best_fit[window]['distrib'] == 'gaussian':
                    self.event_level[window] = ss.norm.ppf(q=1 - self.event_probability[window],\
                                                           loc=self.parameters[window]['loc'].sel(label='with_CC', time=self.event_year),\
                                                           scale=self.parameters[window]['scale'].sel(label='with_CC', time=self.event_year) )

                elif self.best_fit[window]['distrib'] == 'poisson':
                    self.event_level[window] = ss.poisson.ppf(q=1 - self.event_probability[window],\
                                                              loc=self.parameters[window]['loc'].sel(label='with_CC', time=self.event_year),\
                                                              mu=self.parameters[window]['mu'].sel(label='with_CC', time=self.event_year) )

                elif self.best_fit[window]['distrib'] == 'GPD':
                    self.event_level[window] = ss.genpareto.ppf(q=1 - self.event_probability[window],\
                                                                loc=self.parameters[window]['loc'].sel(label='with_CC', time=self.event_year),\
                                                                scale=self.parameters[window]['scale'].sel(label='with_CC', time=self.event_year),\
                                                                c=self.parameters[window]['shape'].sel(label='with_CC', time=self.event_year) )

                elif self.best_fit[window]['distrib'] == 'skewnorm':
                    self.event_level[window] = ss.skewnorm.ppf(q=1 - self.event_probability[window],\
                                                               loc=self.parameters[window]['loc'].sel(label='with_CC', time=self.event_year),\
                                                               scale=self.parameters[window]['scale'].sel(label='with_CC', time=self.event_year),\
                                                               a=self.parameters[window]['shape'].sel(label='with_CC', time=self.event_year) )
                # taking median of this value, instead of full distribution. Oberved to increase ranges for low probability events, though not changing much center of distribution of probabilities.
                self.event_level[window] = np.median(self.event_level[window])
    #--------------------------------------------------------------------------------

    
    
    #--------------------------------------------------------------------------------
    # CALCULATION OF PROBABILITIES
    def full_calc_probas( self, evt_obs, conf_interval=95, distrib_range=95, FAR_or_PR='PR' ):
        # directly saving that, will be used for plots as well if required
        self.evt_obs = evt_obs
        for window in self.list_windows:
            if 'member' in self.evt_obs[window].coords:
                self.evt_obs[window] = self.evt_obs[window].mean('member')
        self.FAR_or_PR = FAR_or_PR
        self.conf_interval = conf_interval
        self.dico_q_confid = {'confid_bottom':(1-self.conf_interval/100)/2, 'confid_upper':1-(1-self.conf_interval/100)/2}
        self.distrib_range = distrib_range
        self.dico_q_range = {'low':(1-self.distrib_range/100)/2, 'high':1-(1-self.distrib_range/100)/2}
        
        # crucial part, defining level of event: direct for reference dataset, but infered from reference for other datasets
        self.eval_event_level()
        
        # calculating probabilities
        self.calc_probas()
        self.calc_intensities()
        
        # deducing attribution metric
        if self.FAR_or_PR == 'FAR':
            self.calc_FAR( block_neg100=True )
        else:
            self.calc_PR()
        self.calc_I()
    
    
    def calc_probas( self, avoid_0=True ):
        # looping on windows
        self.probabilities = { window:xr.Dataset() for window in self.list_windows }
        for window in self.list_windows:
            # calculating probabilities to have this value or below
            if self.best_fit[window]['distrib'] == 'GEV':
                tmp0 = ss.genextreme.cdf(self.event_level[window],\
                                         loc=self.parameters[window]['loc'].sel(time=self.event_year),\
                                         scale=self.parameters[window]['scale'].sel(time=self.event_year),\
                                         c=-self.parameters[window]['shape'].sel(time=self.event_year) )
            elif self.best_fit[window]['distrib'] == 'gaussian':
                tmp0 = ss.norm.cdf(self.event_level[window],\
                                   loc=self.parameters[window]['loc'].sel(time=self.event_year),\
                                   scale=self.parameters[window]['scale'].sel(time=self.event_year) )
            elif self.best_fit[window]['distrib'] == 'poisson':
                tmp0 = ss.poisson.cdf(self.event_level[window],\
                                      loc=self.parameters[window]['loc'].sel(time=self.event_year),\
                                      mu=self.parameters[window]['mu'].sel(time=self.event_year) )
            elif self.best_fit[window]['distrib'] == 'GPD':
                tmp0 = ss.genpareto.cdf(self.event_level[window],\
                                         loc=self.parameters[window]['loc'].sel(time=self.event_year),\
                                         scale=self.parameters[window]['scale'].sel(time=self.event_year),\
                                         c=self.parameters[window]['shape'].sel(time=self.event_year) )
            elif self.best_fit[window]['distrib'] == 'skewnorm':
                tmp0 = ss.skewnorm.cdf(self.event_level[window],\
                                       loc=self.parameters[window]['loc'].sel(time=self.event_year),\
                                       scale=self.parameters[window]['scale'].sel(time=self.event_year),\
                                       a=self.parameters[window]['shape'].sel(time=self.event_year) )
            prob = 1 - tmp0
                
            # some events may be so unlikely without climate change that the proba is 0. However, will cause issues when calculating FAR or PR:
            if avoid_0:
                tmp0[np.where(tmp0 == 0)] = self.min_acceptable_proba
                
            # deduce proba to exceed this value
            self.probabilities[window]['full'] = xr.DataArray( prob, coords={'label':self.parameters[window].label, 'bootstrap':self.parameters[window].bootstrap}, dims=('label', 'bootstrap',) )
            
            # calculating mean, median, confidence interval
            self.probabilities[window]['mean'] = self.probabilities[window]['full'].mean('bootstrap')
            self.probabilities[window]['median'] = self.probabilities[window]['full'].median('bootstrap')
            for q in self.dico_q_confid:
                self.probabilities[window][q] = self.probabilities[window]['full'].quantile(q=self.dico_q_confid[q], dim='bootstrap').drop('quantile')

    
    def calc_intensities( self, avoid_0=True ):
        # looping on windows
        self.intensities = { window:xr.Dataset() for window in self.list_windows }
        for window in self.list_windows:
            level = 1 - self.probabilities[window]['mean'].sel(label='with_CC').values
            
            # calculating probabilities to have this value or below
            if self.best_fit[window]['distrib'] == 'GEV':
                tmp1 = ss.genextreme.ppf(level,\
                                         loc=self.parameters[window]['loc'].sel(time=self.event_year),\
                                         scale=self.parameters[window]['scale'].sel(time=self.event_year),\
                                         c=-self.parameters[window]['shape'].sel(time=self.event_year) )
            elif self.best_fit[window]['distrib'] == 'gaussian':
                tmp1 = ss.norm.ppf(level,\
                                   loc=self.parameters[window]['loc'].sel(time=self.event_year),\
                                   scale=self.parameters[window]['scale'].sel(time=self.event_year) )
            elif self.best_fit[window]['distrib'] == 'poisson':
                tmp1 = ss.poisson.ppf(level,\
                                      loc=self.parameters[window]['loc'].sel(time=self.event_year),\
                                      mu=self.parameters[window]['mu'].sel(time=self.event_year) )
            elif self.best_fit[window]['distrib'] == 'GPD':
                tmp1 = ss.genpareto.ppf(level,\
                                         loc=self.parameters[window]['loc'].sel(time=self.event_year),\
                                         scale=self.parameters[window]['scale'].sel(time=self.event_year),\
                                         c=self.parameters[window]['shape'].sel(time=self.event_year) )
            elif self.best_fit[window]['distrib'] == 'skewnorm':
                tmp1 = ss.skewnorm.ppf(level,\
                                       loc=self.parameters[window]['loc'].sel(time=self.event_year),\
                                       scale=self.parameters[window]['scale'].sel(time=self.event_year),\
                                       a=self.parameters[window]['shape'].sel(time=self.event_year) )
                
            # some events may be so unlikely without climate change that the proba is 0. However, will cause issues when calculating FAR or PR:
            #if avoid_0:
            #    tmp0[np.where(tmp0 == 0)] = self.min_acceptable_proba
                
            # deduce proba to exceed this value
            self.intensities[window]['full'] = xr.DataArray( tmp1, coords={'label':self.parameters[window].label, 'bootstrap':self.parameters[window].bootstrap}, dims=('label', 'bootstrap',) )
            
            # calculating mean, median, confidence interval
            self.intensities[window]['mean'] = self.intensities[window]['full'].mean('bootstrap')
            self.intensities[window]['median'] = self.intensities[window]['full'].median('bootstrap')
            for q in self.dico_q_confid:
                self.intensities[window][q] = self.intensities[window]['full'].quantile(q=self.dico_q_confid[q], dim='bootstrap').drop('quantile')
        
    
    def calc_FAR( self, block_neg100=True):
        # initializing
        self.FAR = { window:xr.Dataset() for window in self.list_windows }
        for window in self.list_windows:
            # calculating values
            vals = 1 - self.probabilities[window]['full'].sel(label='without_CC').drop('label') / self.probabilities[window]['full'].sel(label='with_CC').drop('label')
            if block_neg100:
                vals = xr.where( vals<-1, -1, vals )
            self.FAR[window]['values'] = vals

            # calculating mean & median
            self.FAR[window]['mean'] = vals.mean('bootstrap')
            self.FAR[window]['median'] = vals.median('bootstrap')

            # calculating confidence interval
            for q in self.dico_q_confid:
                self.FAR[window][q] = vals.quantile(q=self.dico_q_confid[q], dim='bootstrap')
    
    def calc_PR( self ):
        # initializing
        self.PR = { window:xr.Dataset() for window in self.list_windows }
        for window in self.list_windows:
            # calculating values
            vals = self.probabilities[window]['full'].sel(label='with_CC').drop('label') / self.probabilities[window]['full'].sel(label='without_CC').drop('label')
            self.PR[window]['values'] = vals

            # calculating mean & median
            #vals = xr.where( (vals.values > self.limit_PR_WWA), self.limit_PR_WWA, vals ) # blocking very high PR cf WWA approach: only for analysis, plots & interpretation, but not during detailed calculations
            self.PR[window]['mean'] = vals.mean('bootstrap')
            self.PR[window]['median'] = vals.median('bootstrap')

            # calculating confidence interval
            for q in self.dico_q_confid:
                self.PR[window][q] = vals.quantile(q=self.dico_q_confid[q], dim='bootstrap')

    def calc_I( self ):
        # initializing
        self.I = { window:xr.Dataset() for window in self.list_windows }
        for window in self.list_windows:
            # calculating values
            vals = self.intensities[window]['full'].sel(label='with_CC').drop('label') - self.intensities[window]['full'].sel(label='without_CC').drop('label')
            self.I[window]['values'] = vals

            # calculating mean & median
            #vals = xr.where( (vals.values > self.limit_PR_WWA), self.limit_PR_WWA, vals ) # blocking very high PR cf WWA approach: only for analysis, plots & interpretation, but not during detailed calculations
            self.I[window]['mean'] = vals.mean('bootstrap')
            self.I[window]['median'] = vals.median('bootstrap')

            # calculating confidence interval
            for q in self.dico_q_confid:
                self.I[window][q] = vals.quantile(q=self.dico_q_confid[q], dim='bootstrap')

    
    @staticmethod
    def g(k, shape):
        return gamma(1 - k * shape)

    def GEV_mean( self, loc, scale, shape):
        # case: shape < 1
        out = loc + scale * (self.g(1,shape)  - 1) / shape
        # case: shape > 1
        out[np.where(shape > 1)] = np.inf
        # case: shape ~ 0
        ind = np.where(np.isclose(shape,0))
        out[ind] = (loc + scale * e)[ind]
        return out

    def fct_central_range_distr( self ):
        # initialize
        evol_mean = { window:xr.Dataset() for window in self.list_windows }
        evol_range = { window:{q:xr.Dataset() for q in self.dico_q_range} for window in self.list_windows }
        for window in self.list_windows:
            # Calculating all required info
            if self.best_fit[window]['distrib'] == 'GEV':
                # warning with c=-shape
                mm = ss.genextreme.median( loc=self.parameters[window]['loc'], scale=self.parameters[window]['scale'], c=-self.parameters[window]['shape'] )
                rr = {q:ss.genextreme.ppf( q=self.dico_q_range[q], loc=self.parameters[window]['loc'], scale=self.parameters[window]['scale'], c=-self.parameters[window]['shape'] ) for q in self.dico_q_range}

            elif self.best_fit[window]['distrib'] == 'gaussian':
                mm = ss.norm.median( loc=self.parameters[window]['loc'], scale=self.parameters[window]['scale'] )
                rr = {q:ss.norm.ppf( q=self.dico_q_range[q], loc=self.parameters[window]['loc'], scale=self.parameters[window]['scale'] ) for q in self.dico_q_range}

            elif self.best_fit[window]['distrib'] == 'poisson':
                # warning with mu, not scale
                # warning with logpmf, not logpdf
                mm = ss.poisson.median( loc=self.parameters[window]['loc'], mu=self.parameters[window]['mu'] )
                rr = {q:ss.poisson.ppf( q=self.dico_q_range[q], loc=self.parameters[window]['loc'], mu=self.parameters[window]['mu'] ) for q in self.dico_q_range}

            elif self.best_fit[window]['distrib'] == 'GPD':
                mm = ss.genpareto.median( loc=self.parameters[window]['loc'], scale=self.parameters[window]['scale'], c=self.parameters[window]['shape'] )
                rr = {q:ss.genpareto.ppf( q=self.dico_q_range[q], loc=self.parameters[window]['loc'], scale=self.parameters[window]['scale'], c=self.parameters[window]['shape'] ) for q in self.dico_q_range}

            elif self.best_fit[window]['distrib'] == 'skewnorm':
                mm = ss.skewnorm.median( loc=self.parameters[window]['loc'], scale=self.parameters[window]['scale'], a=self.parameters[window]['shape'] )
                rr = {q:ss.skewnorm.ppf( q=self.dico_q_range[q], loc=self.parameters[window]['loc'], scale=self.parameters[window]['scale'], a=self.parameters[window]['shape'] ) for q in self.dico_q_range}

            # archiving them
            crds= {'label':self.parameters[window].label, 'bootstrap':self.parameters[window].bootstrap, 'time':self.parameters[window].time}
            evol_mean[window]['full'] = xr.DataArray( mm, coords=crds, dims=('label', 'bootstrap', 'time',) )
            for q in self.dico_q_range:
                evol_range[window][q]['full'] = xr.DataArray( rr[q], coords=crds, dims=('label', 'bootstrap', 'time',) )
            
            # summarizing by calculating confidence intervals
            evol_mean[window]['mean'] = evol_mean[window]['full'].mean('bootstrap')
            evol_mean[window]['median'] = evol_mean[window]['full'].median('bootstrap')
            for q in self.dico_q_confid:
                evol_mean[window][q] = evol_mean[window]['full'].quantile(q=self.dico_q_confid[q], dim='bootstrap')
            for p in self.dico_q_range:
                evol_range[window][p]['mean'] = evol_range[window][p]['full'].mean('bootstrap')
                evol_range[window][p]['median'] = evol_range[window][p]['full'].median('bootstrap')
                for q in self.dico_q_confid:
                    evol_range[window][p][q] = evol_range[window][p]['full'].quantile(q=self.dico_q_confid[q], dim='bootstrap')

        return evol_mean, evol_range

    def eval_quantiles_obs( self ):
        self.quantiles_obs = {window:xr.Dataset() for window in self.list_windows}
        for window in self.list_windows:
            years = [yr for yr in self.evt_obs[window].time.values if yr in self.parameters[window]['loc'].time]
            
            # Calculating the probabilities of the observations
            if self.best_fit[window]['distrib'] == 'GEV':
                # warning with c=-shape
                tmp = ss.genextreme.cdf(x=self.evt_obs[window].sel(time=years),\
                                        loc=self.parameters[window]['loc'].sel(label='with_CC', time=years),\
                                        scale=self.parameters[window]['scale'].sel(label='with_CC', time=years),\
                                        c=-self.parameters[window]['shape'].sel(label='with_CC', time=years) )

            elif self.best_fit[window]['distrib'] == 'gaussian':
                tmp = ss.norm.cdf(x=self.evt_obs[window].sel(time=years),\
                                  loc=self.parameters[window]['loc'].sel(label='with_CC', time=years),\
                                  scale=self.parameters[window]['scale'].sel(label='with_CC', time=years) )

            elif self.best_fit[window]['distrib'] == 'poisson':
                tmp = ss.poisson.cdf(x=self.evt_obs[window].sel(time=years),\
                                     loc=self.parameters[window]['loc'].sel(label='with_CC', time=years),\
                                     mu=self.parameters[window]['mu'].sel(label='with_CC', time=years) )

            elif self.best_fit[window]['distrib'] == 'GPD':
                tmp = ss.genpareto.cdf(x=self.evt_obs[window].sel(time=years),\
                                       loc=self.parameters[window]['loc'].sel(label='with_CC', time=years),\
                                       scale=self.parameters[window]['scale'].sel(label='with_CC', time=years),\
                                       c=self.parameters[window]['shape'].sel(label='with_CC', time=years) )

            elif self.best_fit[window]['distrib'] == 'skewnorm':
                tmp = ss.skewnorm.cdf(x=self.evt_obs[window].sel(time=years),\
                                      loc=self.parameters[window]['loc'].sel(label='with_CC', time=years),\
                                      scale=self.parameters[window]['scale'].sel(label='with_CC', time=years),\
                                      a=self.parameters[window]['shape'].sel(label='with_CC', time=years) )
                
            # deducing the quantiles of such probabilities for a standard normal distribution
            self.quantiles_obs[window]['full'] = xr.DataArray( ss.norm.ppf(q=tmp, loc=0, scale=1), coords={'bootstrap':self.parameters[window].bootstrap, 'time':years}, dims=('bootstrap', 'time',) )
            self.quantiles_obs[window]['mean'] = self.quantiles_obs[window]['full'].mean('bootstrap')
            self.quantiles_obs[window]['median'] = self.quantiles_obs[window]['full'].median('bootstrap')
            for q in self.dico_q_confid:
                self.quantiles_obs[window][q] = self.quantiles_obs[window]['full'].quantile(q=self.dico_q_confid[q], dim='bootstrap').drop('quantile')
        
        
    def fct_pdf_distr( self, x ):
        # preparing
        pdf = {window:xr.Dataset() for window in self.list_windows}
        x_xr = xr.DataArray( x, coords={'bins_event':x}, dims=('bins_event') )
        
        # creating this new coordinate
        for window in self.list_windows:
            crds = {'label':self.parameters[window].label, 'bootstrap':self.parameters[window].bootstrap, 'bins_event':x }
            dims = ('label', 'bootstrap', 'bins_event',)
            
            # Calculating all required info
            if self.best_fit[window]['distrib'] == 'GEV':
                # warning with c=-shape
                tmp = ss.genextreme.pdf(x=x_xr,\
                                        loc=self.parameters[window]['loc'].sel(time=self.event_year).expand_dims('bins_event', axis=-1),\
                                        scale=self.parameters[window]['scale'].sel(time=self.event_year).expand_dims('bins_event', axis=-1),\
                                        c=-self.parameters[window]['shape'].sel(time=self.event_year).expand_dims('bins_event', axis=-1) )

            elif self.best_fit[window]['distrib'] == 'gaussian':
                tmp = ss.norm.pdf(x=x_xr,\
                                  loc=self.parameters[window]['loc'].sel(time=self.event_year).expand_dims('bins_event', axis=-1),\
                                  scale=self.parameters[window]['scale'].sel(time=self.event_year).expand_dims('bins_event', axis=-1))

            elif self.best_fit[window]['distrib'] == 'poisson':
                # warning with mu, not scale
                # warning with logpmf, not logpdf
                tmp = ss.poisson.pdf(x=x_xr,\
                                     loc=self.parameters[window]['loc'].sel(time=self.event_year).expand_dims('bins_event', axis=-1),\
                                     mu=self.parameters[window]['mu'].sel(time=self.event_year).expand_dims('bins_event', axis=-1))

            elif self.best_fit[window]['distrib'] == 'GPD':
                tmp = ss.genpareto.pdf(x=x_xr,\
                                       loc=self.parameters[window]['loc'].sel(time=self.event_year).expand_dims('bins_event', axis=-1),\
                                       scale=self.parameters[window]['scale'].sel(time=self.event_year).expand_dims('bins_event', axis=-1),\
                                       c=self.parameters[window]['shape'].sel(time=year).expand_dims('bins_event', axis=-1) )

            elif self.best_fit[window]['distrib'] == 'skewnorm':
                tmp = ss.skewnorm.pdf(x=x_xr,\
                                      loc=self.parameters[window]['loc'].sel(time=year).expand_dims('bins_event', axis=-1),\
                                      scale=self.parameters[window]['scale'].sel(time=year).expand_dims('bins_event', axis=-1),\
                                      a=self.parameters[window]['shape'].sel(time=self.event_year).expand_dims('bins_event', axis=-1) )

            # shaping these pdf
            pdf[window]['full'] = xr.DataArray( tmp, coords=crds, dims=dims  )

            # summarizing by calculating confidence intervals
            pdf[window]['mean'] = pdf[window]['full'].mean('bootstrap')
            pdf[window]['median'] = pdf[window]['full'].median('bootstrap')
            for q in self.dico_q_confid:
                pdf[window][q] = pdf[window]['full'].quantile(q=self.dico_q_confid[q], dim='bootstrap')
        return pdf
    #--------------------------------------------------------------------------------
    
    
    #--------------------------------------------------------------------------------
    # PLOT
    def plot_QQ( self, ax, window, colors ):
        percentiles = np.arange( 0.5, 99.5+0.1, 0.5 )
        list_percentiles_markers = np.array([1, 2.5, 5, 25, 50, 75, 95, 97.5, 99])
        if np.any( [p not in percentiles for p in list_percentiles_markers] ):
            raise Exception( 'will miss some percentile markers' )
            
        # "quantiles of fit"
        Q_fit = ss.norm.ppf(q=percentiles/100, loc=0, scale=1)
        Q_fit_markers = ss.norm.ppf(q=list_percentiles_markers/100, loc=0, scale=1)

        # quantiles of observations
        if True:
            Q_obs = {'full': self.quantiles_obs[window]['full'].quantile(q=percentiles/100, dim='time').rename({'quantile':'percentile'})}
            Q_obs['full'].coords['percentile'] = percentiles
            Q_obs['mean'] = Q_obs['full'].mean('bootstrap')
            Q_obs['median'] = Q_obs['full'].median('bootstrap')
            for q in self.dico_q_confid:
                Q_obs[q] = Q_obs['full'].quantile(q=self.dico_q_confid[q], dim='bootstrap')
        else:
            Q_obs = self.quantiles_obs[window].quantile(q=percentiles/100, dim='time').rename({'quantile':'percentile'})
            Q_obs.coords['percentile'] = percentiles
        
        # lines
        plt.plot( Q_obs['median'], Q_fit, color=colors['fit_with_CC'], lw=2, label='Fit: averaged median' )
        xl = plt.xlim()
        plt.plot( np.linspace(xl[0], xl[1], 2), np.linspace(xl[0], xl[1], 2), lw=2, color='black', zorder=0, ls=':', label='identity line')
        plt.fill_betweenx( x1=Q_obs['confid_bottom'], x2=Q_obs['confid_upper'], y=Q_fit, facecolor=colors['fit_with_CC'], alpha=0.3, label='Fit: averaged '+str(self.distrib_range)+'% range' )
        # markers
        plt.scatter( Q_obs['median'].sel(percentile=list_percentiles_markers), Q_fit_markers, marker='o', color='black', s=self.fontsizes['marker'], zorder=10 )
        for i, p in enumerate(list_percentiles_markers):
            if np.isinf(Q_obs['median'].sel(percentile=p))==False:
                plt.text(x=Q_obs['median'].sel(percentile=p).values, y=Q_fit_markers[i], s=str(p)+'%',\
                         fontdict={"size":self.fontsizes['text_quantiles']}, color="k", va="bottom", ha="center", rotation="horizontal", rotation_mode="anchor")

        # polishing
        plt.grid()
        _ = plt.xticks( size=self.fontsizes['ticks'] )
        _ = plt.yticks( size=self.fontsizes['ticks'] )
        plt.xlabel( 'Quantiles: sample ('+ u'\u00B0C'+')', size=self.fontsizes['label'] )
        plt.ylabel( 'Quantiles: best fit of $T_{'+str(window)+'}$ ('+ u'\u00B0C'+')', size=self.fontsizes['label'] )
        plt.title( 'Comparison of quantiles', size=self.fontsizes['title'] )
        lgd = plt.legend( loc='center', bbox_to_anchor=(0.5,-0.175), ncol=2, prop={'size':self.fontsizes['legend']} )
        return ax

    def plot_obs_fit( self, ax, window, yr_plot_start, yr_plot_end, evol_mean, evol_range, yl_tsr, colors, zorders, ls_event ):
        years_plot = [yr for yr in self.evt_obs[window]['time'] if (yr_plot_start <= yr) and (yr <= yr_plot_end)]
        ttl = self.write_fit(self.best_fit[window], window)
        plt.title( 'Best fit: '+ttl, fontsize=self.fontsizes['title'] )

        # ploting observations
        if self.name_data == self.name_reference:
            plot_obs = plt.plot(self.evt_obs[window].time.sel( time=years_plot ), self.evt_obs[window].sel( time=years_plot ).values, \
                                lw=2, color=colors['obs'], label='Observations: ' + str(np.round(self.event_level[window], 2)) + u'\u00B0C'+' at event', zorder=zorders['obs'] )
        else:
            plot_obs = plt.plot(self.evt_obs[window].time.sel( time=years_plot ), self.evt_obs[window].sel( time=years_plot ).values, \
                                lw=2, color=colors['obs'], zorder=zorders['obs'] )

        # calculating evolutions under the parameters
        plot_2nd_legend = []
        for CC in ['with', 'without']:
            # ploting fit: median
            plot = plt.plot(self.evt_obs[window].time.sel( time=years_plot ), evol_mean[window]['median'].sel(label=CC+'_CC', time=years_plot),\
                            color=colors['fit_'+CC+'_CC'], lw=2, label='Fit: averaged median, '+CC+' CC', zorder=zorders['fit_'+CC+'_CC'] )
            plot_2nd_legend.append( plot[0] )
            for q in self.dico_q_confid:
                plt.plot(self.evt_obs[window].time.sel( time=years_plot ), evol_mean[window][q].sel(label=CC+'_CC', time=years_plot),\
                         color=colors['fit_'+CC+'_CC'], lw=1, ls='--', zorder=zorders['fit_'+CC+'_CC'] )

            # ploting fit: range
            plot = plt.fill_between(self.evt_obs[window].time.sel( time=years_plot ),\
                                    evol_range[window]['low']['median'].sel(label=CC+'_CC', time=years_plot), evol_range[window]['high']['median'].sel(label=CC+'_CC', time=years_plot), \
                                    facecolor=colors['fit_'+CC+'_CC'], alpha=0.3, label='Fit: averaged '+str(self.distrib_range)+'% range, '+CC+' CC', zorder=zorders['fit_'+CC+'_CC'] )
            plot_2nd_legend.append( plot )
            for p in ['low', 'high']:
                for q in self.dico_q_confid:
                    plt.plot(self.evt_obs[window].time.sel( time=years_plot ), evol_range[window][p][q].sel(label=CC+'_CC', time=years_plot),\
                             color=colors['fit_'+CC+'_CC'], lw=1, ls='--', zorder=zorders['fit_'+CC+'_CC'] )

        # ploting event
        plt.vlines( x=[self.event_year], ymin=yl_tsr[0], ymax=yl_tsr[1], color=colors['event'], lw=3, ls=ls_event, zorder=zorders['event'] )

        # polishing
        plt.xlim( yr_plot_start, yr_plot_end )
        plt.xlabel( 'Time', size=self.fontsizes['label'] )
        plt.ylabel( '$T_{'+str(window)+'}$ ('+ u'\u00B0C'+')', size=self.fontsizes['label'], labelpad=-0.1 )
        plt.grid()
        if self.name_data == self.name_reference:
            # first legend: only observations
            lgd1 = plt.legend( plot_obs, [plot_obs[0].get_label()], loc='upper left', prop={'size':self.fontsizes['legend']} )
            ax.add_artist(lgd1)
        # second legend: description of fits
        if window == self.list_windows[-1]:
            inds = [0,2,1,3]
            lgd2 = plt.legend( [plot_2nd_legend[i] for i in inds], [plot_2nd_legend[i].get_label() for i in inds], loc='center', bbox_to_anchor=(0.5,-0.175), ncol=2, prop={'size':self.fontsizes['legend']} )
        if window == self.list_windows[-1]:
            ax.add_artist(lgd2)

        _ = plt.xticks( size=self.fontsizes['ticks'] )
        _ = plt.yticks( size=self.fontsizes['ticks'] )
        if window != self.list_windows[-1]:
            ax.tick_params(axis='x',label1On=False)
        plt.ylim( yl_tsr[0], yl_tsr[1] )
        return ax
    
    def plot_proba( self, ax, window, yl_tsr,  n_bins, colors, zorders, hatches, ls_event ):
        if window == self.list_windows[0]:
            plt.title( 'Probabilities of the event', fontsize=self.fontsizes['title'] )

        # preparing
        range_event = np.linspace(yl_tsr[0], yl_tsr[1], n_bins)
        pdf = self.fct_pdf_distr( x=range_event )
        
        # ploting distributions
        plot_1st_legend, plot_2nd_legend = [], []
        for CC in ['with', 'without']:
            proba_median = np.round( 100. * self.probabilities[window]['median'].sel(label=CC+'_CC').values, 2)
            proba_pct_bottom = np.round( 100. * self.probabilities[window]['confid_bottom'].sel(label=CC+'_CC').values, 2)
            proba_pct_upper = np.round( 100. * self.probabilities[window]['confid_upper'].sel(label=CC+'_CC').values, 2)

            # ploting fit: mean
            plot = plt.plot( pdf[window]['median'].sel(label=CC+'_CC'), range_event, color=colors['fit_'+CC+'_CC'], lw=2, label='Fit: averaged PDF, '+CC+' CC', zorder=zorders['fit_'+CC+'_CC'] )
            plot_2nd_legend.append( plot[0] )
            for q in self.dico_q_confid:
                plt.plot( pdf[window][q].sel(label=CC+'_CC'), range_event, color=colors['fit_'+CC+'_CC'], lw=1, ls='--', zorder=zorders['fit_'+CC+'_CC'] )
                
            # ploting probas
            ii = np.argmin( np.abs( range_event - self.event_level[window] ) )
            plot = plt.fill_betweenx(y=range_event[ii:], x1=pdf[window]['median'].sel(label=CC+'_CC').values[ii:], x2=0, color=None, alpha=0, lw=2, hatch=hatches['fit_'+CC+'_CC'],\
                                     label='Proba. '+CC+' CC: '+str(proba_median)+'% ['+str(proba_pct_bottom)+'; '+str(proba_pct_upper)+']', zorder=zorders['fit_'+CC+'_CC'] )
            plot_1st_legend.append( plot )

        # ploting event
        xl = (0, ax.get_xlim()[1])
        if self.name_data == self.name_reference:
            plot_event = plt.hlines(y=[np.round(self.event_level[window], 2)], xmin=xl[0], xmax=xl[1], color=colors['event'], lw=3, ls=ls_event,\
                                    label='Event: '+str(np.round(self.event_level[window], 2)) + u'\u00B0C', zorder=zorders['event'] )
        else:
            plot_event = plt.hlines(y=[np.round(self.event_level[window], 2)], xmin=xl[0], xmax=xl[1], color=colors['event'], lw=3, ls=ls_event, zorder=zorders['event'] )
        plot_1st_legend.append( plot_event )

        # polishing
        plt.xlim( self.evt_obs[window].time.values[0], self.evt_obs[window].time.values[-1] )
        plt.ylabel( '$T_{'+str(window)+'}$ ('+ u'\u00B0C'+')', size=self.fontsizes['label'] )
        plt.grid()

        # first legend: event & probas
        inds = [2,0,1]
        lgd1 = plt.legend( [plot_1st_legend[i] for i in inds], [plot_1st_legend[i].get_label() for i in inds], loc='lower right', prop={'size':self.fontsizes['legend']} )
        # second legend: description of fits
        if window == self.list_windows[-1]:
            lgd2 = plt.legend( plot_2nd_legend, [plot.get_label() for plot in plot_2nd_legend], loc='center', bbox_to_anchor=(0.5,-0.175), prop={'size':self.fontsizes['legend']} )
        ax.add_artist(lgd1)
        if window == self.list_windows[-1]:
            ax.add_artist(lgd2)

        ax.tick_params(axis='x',label1On=False)
        _ = plt.yticks( size=self.fontsizes['ticks'] )
        if window == self.list_windows[-1]:
            plt.xlabel('Density of probability', size=self.fontsizes['label'])
        plt.xlim( xl[0], xl[1] )
        plt.ylim( yl_tsr[0], yl_tsr[1] )
        return ax
    
    def plot_attrib_metric( self, ax, window, attrib_metric, xmin_attrib_metric, xmax_attrib_metric, zorders, colors ):

        if window == self.list_windows[0]:
            if self.FAR_or_PR == 'FAR':
                plt.title( 'Fraction of\nattributable risk', fontsize=self.fontsizes['title'] )
            else:
                plt.title( 'Probability Ratio', fontsize=self.fontsizes['title'] )

        # preparing histogram
        if self.FAR_or_PR == 'FAR':
            _ = plt.hist( 100 * attrib_metric[window]['values'] , density=True, bins=np.linspace(xmin_attrib_metric, xmax_attrib_metric, 50+1), color='grey', alpha=0.75 )
        else:
            _ = plt.hist( attrib_metric[window]['values'] , density=True, bins=np.logspace(np.log10(xmin_attrib_metric), np.log10(xmax_attrib_metric), 50+1), color='grey', alpha=0.75 )
            ax.set_xscale('log')

        # adding values
        yl = ax.get_ylim()
        if self.FAR_or_PR == 'FAR':
            lbl = 'FAR:\n'+str(np.round(100*attrib_metric[window]['median'],2))+'% [' + str(np.round(100*attrib_metric[window]['confid_bottom'],2)) + '; '+str(np.round(100*attrib_metric[window]['confid_upper'],2)) + ']'
            plt.vlines( x=[100 * attrib_metric[window]['median']], ymin=yl[0], ymax=yl[1], color=colors['event'], lw=3, ls='-', zorder=zorders['event'], label=lbl )
            for q in self.dico_q_confid:
                plt.vlines( x=[100 * attrib_metric[window][q]], ymin=yl[0], ymax=yl[1], color=colors['event'], lw=2, ls='--', zorder=zorders['event'] )
        else:
            lbl = 'PR:\n'+self.sci_notation(attrib_metric[window]['median'], sig_fig=2)+' [' + self.sci_notation(attrib_metric[window]['confid_bottom'], sig_fig=2) + '; ' + self.sci_notation(attrib_metric[window]['confid_upper'], sig_fig=2) + ']'
            plt.vlines( x=[attrib_metric[window]['median']], ymin=yl[0], ymax=yl[1], color=colors['event'], lw=3, ls='-', zorder=zorders['event'], label=lbl )
            for q in self.dico_q_confid:
                plt.vlines( x=[attrib_metric[window][q]], ymin=yl[0], ymax=yl[1], color=colors['event'], lw=2, ls='--', zorder=zorders['event'] )

        _ = plt.xticks( size=self.fontsizes['ticks'] )# "rotation" here applies only to major ticks
        ax.tick_params(axis='x',which='both', labelrotation=-45)
        ax.tick_params(axis='y',label1On=False)
        plt.grid()
        plt.xlim( xmin_attrib_metric, xmax_attrib_metric )
        plt.ylim( yl )
        lgd = plt.legend( loc='upper left', prop={'size':self.fontsizes['legend']} )
        lgd.set_zorder( np.max(list(zorders.values()))+1 )
        #lgd.get_frame().set_alpha(0.9)# instead of 0.8
        #lgd.get_frame().set_facecolor((1.0, 1.0, 1.0, 0.9))# instead of (1.0, 1.0, 1.0, 0.8)
        if self.FAR_or_PR == 'FAR':
            plt.xlabel('FAR (%)', size=self.fontsizes['label'])
            plt.ylabel('Density of FAR', size=self.fontsizes['label'])
        else:
            plt.xlabel('PR', size=self.fontsizes['label'])
            plt.ylabel('Density of PR', size=self.fontsizes['label'])
        return ax
    
    def plot_full(self, plot_start_year, fontsize=16, width=25, n_bins=100, ls_event=':', \
                  colors={'obs':'black', 'fit_with_CC':CB_color_cycle[0], 'fit_without_CC':CB_color_cycle[8], 'event':'black'},\
                  zorders={'obs':20, 'fit_with_CC':10, 'fit_without_CC':0, 'event':20},\
                  hatches = {'fit_with_CC':'///', 'fit_without_CC':'\\\\\\'}, close_fig=True ):#{'fit_with_CC':'///', 'fit_without_CC':'...'}
        # general preparation
        self.fontsizes = {'ticks':0.8*fontsize, 'legend':0.7*fontsize, 'label':0.9*fontsize, 'title':fontsize, 'marker':25*fontsize/20, 'text_quantiles':0.9*fontsize}
        win0 = self.list_windows[0]
        yr_plot_start = plot_start_year
        yr_plot_end = np.min( [np.max(self.evt_obs[win0].time.values), np.max(self.parameters[win0].time.values)] )

        # preparing some values: for plot of probabilities and attrib metric
        if self.FAR_or_PR == 'FAR':
            attrib_metric = self.FAR
            xmin_attrib_metric = np.min( [np.percentile(a=100*attrib_metric[window]['values'], q=0.25 ) for window in self.list_windows] )
            xmax_attrib_metric = np.max( [np.max(100*attrib_metric[window]['values']) for window in self.list_windows] )
        else:
            attrib_metric = {window: self.PR[window].copy() for window in self.list_windows}
            # blocking very high PR cf WWA approach: only for analysis, plots & interpretation, but not during calculations
            for window in self.list_windows:
                attrib_metric[window]['values'] = xr.where( (attrib_metric[window]['values'] > self.limit_PR_WWA), self.limit_PR_WWA, attrib_metric[window]['values'] )
            xmin_attrib_metric = 1.e-4
            xmax_attrib_metric = 1.e4 # : not related to limit_PR_WWA
    
        # calculate probability of observations, used for PP plot
        self.eval_quantiles_obs()
            
        # preparing values for PP-plot and obs & fit
        evol_mean, evol_range = self.fct_central_range_distr()
        tmp_min = [evol_range[window]['low']['confid_bottom'].min('label') for window in self.list_windows]
        tmp_max = [evol_range[window]['high']['confid_upper'].max('label') for window in self.list_windows]
        yl_tsr = [np.min(tmp_min) - 0.05*(np.max(tmp_max)-np.min(tmp_min)), np.max(tmp_max) + 0.05*(np.max(tmp_max)-np.min(tmp_min)) ]
        
        # preparing plot
        fig = plt.figure( figsize=(width, width*0.25*len(self.list_windows)) )
        spec = gridspec.GridSpec(nrows=len(self.list_windows), ncols=4, wspace=0.2 , figure=fig, width_ratios=[25, 37.5, 22.5, 15] )

        # looping on windows
        for window in self.list_windows:
            #----------------------------------------
            # PLOT: PP-plot
            ax = plt.subplot( spec[self.list_windows.index(window),0] )
            ax = self.plot_QQ( ax, window, colors )
            
            #----------------------------------------
            # PLOT: observations & fit
            ax = plt.subplot( spec[self.list_windows.index(window),1] )
            ax = self.plot_obs_fit( ax, window, yr_plot_start, yr_plot_end, evol_mean, evol_range, yl_tsr, colors, zorders, ls_event )
            
            #----------------------------------------
            # PLOT: probabilities at time event
            ax = plt.subplot( spec[self.list_windows.index(window),2] )
            ax = self.plot_proba( ax, window, yl_tsr, n_bins, colors, zorders, hatches, ls_event )
            
            #----------------------------------------
            # PLOT: Fraction of Attributable Risk  or  Probability Ratio
            ax = plt.subplot( spec[self.list_windows.index(window),3] )
            ax = self.plot_attrib_metric( ax, window, attrib_metric, xmin_attrib_metric, xmax_attrib_metric, zorders, colors )

        # save
        fig_name0 = self.name_file_figure_fit(self.list_windows[0])
        fig.savefig( fig_name0 + '.png', dpi=300 )
        fig.savefig( fig_name0 + '.pdf' )
        if close_fig:
            plt.close(fig)
        return fig
    #--------------------------------------------------------------------------------
    
    
    
    
    #--------------------------------------------------------------------------------
    # WRITE EXPRESSIONS
    def write_fit( self, fit, window ):

        # gathering type of parameters
        categ_params = {}
        for param in fit['parameters']:
            for word in ['loc', 'scale', 'shape', 'mu']:
                if param[:len(word)] == word:
                    if word not in categ_params:
                        categ_params[word] = []
                    categ_params[word].append( param )

        # initialize line
        self.dico_name_covars = {'GMT':' \Delta T_{t}', 'GMTm1':' \Delta T_{t-1}'}

        distrib_written = {'GEV':'GEV', 'poisson':'Poisson', 'gaussian':'\mathcal{N}', 'GPD':'GPD', 'skewnorm':'S\mathcal{N}'}[ fit['distrib'] ]
        self.tmp_expression = '$T_{'+str(window)+', t} \sim ' + distrib_written + '\\left('# \\;
        for param in categ_params: # loc... ; scale.... ; shape... ; mu....
            # identification
            self.written_param = {'loc':'\\mu', 'scale':'\\sigma', 'shape':'\\xi', 'mu':'\\lambda'}[param]

            # constant term
            self.tmp_expression += self.written_param+'_0'
            self.count = 0

            # linear terms
            self.write_linear( params=[p for p in categ_params[param] if 'linear' in p] )

            # power terms
            self.write_power( params=[p for p in categ_params[param] if 'power' in p] )

            # logistic terms
            self.write_logistic( params=[p for p in categ_params[param] if 'logistic' in p] )

            # closure over this term
            last_param_distrib = self.distrib_params[ fit['distrib'] ][-1]
            if param != last_param_distrib:
                self.tmp_expression += ', \\; '
            else:
                self.tmp_expression += '\\right)$'#\\;
                
        return self.tmp_expression


    def write_linear( self, params ):
        for par in params:
            if par[-len('_0'):] != '_0':# making sure that it is not the constant term
                # equation
                self.count += 1
                inp_written = self.dico_name_covars[par.split('_')[2]]
                self.tmp_expression += '+'+self.written_param+'_{'+str(self.count)+'} '+inp_written


    def write_power( self, params ):
        for par in params:
            if par[-len('_0'):] != '_0':# making sure that it is not the constant term
                # equation
                self.count += 1
                inp_written = self.dico_name_covars[par.split('_')[2]]
                pw = par.split('_')[1][len('power'):]
                self.tmp_expression += '+'+self.written_param+'_{'+str(self.count)+'} '+inp_written+'^{'+str(pw)+'}'


    def write_logistic( self, params ):
        # checking that wont do an empty loop but also an empty logistic term
        if len(params) > 0:

            # preparation common to all logistic terms
            left = self.written_param+'_{L}'
            right = self.written_param+'_{R}'
            eps = self.written_param+'_{\\epsilon}'
            tt,ll = '',0

            for par in params:
                if par[-len('_0'):] != '_0':# making sure that it is not the constant term
                    # preparing
                    inp_written = self.dico_name_covars[par.split('_')[2]]
                    self.count += 1

                    # equation
                    lamb = self.written_param+'_{\\lambda,'+str(self.count)+'}'
                    tt += lamb+''+inp_written
                    if cov != inds[-1]:
                        tt += '+'

            self.tmp_expression += '+'+left+'+\\frac{'+right+'-'+left+'}{1+e^{'+tt+'-'+eps+'}}'    
            
    @staticmethod
    def sci_notation(number, sig_fig=2):
        if np.isnan(number) or np.isinf(number):
            return "$+\infty$"
        else:
            ret_string = "{0:.{1:d}e}".format(number, sig_fig)
            a, b = ret_string.split("e")
            # remove leading "+" and strip leading zeros
            b = int(b)
            if b == 0:
                return "$" + a + "$"
            else:
                return "$" + a + ".10^{" + str(b) + "}$"
    #--------------------------------------------------------------------------------
    
    
    
    #--------------------------------------------------------------------------------
    def name_file_parameters(self, window):
        # checking path
        path_save = os.path.join(self.path_results, self.identifier_event)
        if not os.path.exists(path_save):os.makedirs(path_save)
            
        # preparing file
        basis = 'parameters_' + str(self.identifier_event)
        info_data = self.name_data + '-' + window
        info_years = str(self.training_start_year) + '-' + str(self.training_end_year) + '-w' + self.option_train_wo_event*'o' + 'evt'
        info_training = self.weighted_NLL*'weighted' + 'NLL' + '-selected' + self.select_BIC_or_NLL
        return os.path.join(path_save, basis + '_' + info_data + '_' + info_years + '_' + info_training + '.nc' )
    
    def name_file_figure_fit(self, window):
        # checking path
        path_save = os.path.join(self.path_figures, self.identifier_event)
        if not os.path.exists(path_save):os.makedirs(path_save)
            
        # preparing file
        basis = 'figure_' + str(self.identifier_event)
        info_data = self.name_data + '-' + window
        info_years = str(self.training_start_year) + '-' + str(self.training_end_year) + '-w' + self.option_train_wo_event*'o' + 'evt'
        info_training = self.weighted_NLL*'weighted' + 'NLL' + '-selected' + self.select_BIC_or_NLL
        return os.path.join(path_save, basis + '_' + info_data + '_' + info_years + '_' + info_training + '_fit' )

    def name_file_figure_tree(self, window):
        # checking path
        path_save = os.path.join(self.path_figures, self.identifier_event)
        if not os.path.exists(path_save):os.makedirs(path_save)
            
        # preparing file
        basis = 'figure_' + str(self.identifier_event)
        info_data = self.name_data + '-' + window
        info_years = str(self.training_start_year) + '-' + str(self.training_end_year) + '-w' + self.option_train_wo_event*'o' + 'evt'
        info_training = self.weighted_NLL*'weighted' + 'NLL' + '-selected' + self.select_BIC_or_NLL
        return os.path.join(path_save, basis + '_' + info_data + '_' + info_years + '_' + info_training + '_tree' )
    
    def save_parameters(self):
        for window in self.list_windows:
            pars = list( self.best_fit[window]['parameters'].keys() )
            boot = np.arange( self.n_iterations_BS )
            
            # preparing dataset
            OUT = xr.Dataset()
            OUT.coords['parameters'] = pars
            OUT.coords['bootstrap'] = boot
        
            # best fit
            OUT['parameters_best_fit'] = xr.DataArray( [self.best_fit[window]['parameters'][p] for p in pars], dims=('parameters'), coords={'parameters':pars} )
            for item in ['distrib', 'items_fit', 'NLL', 'BIC']:
                OUT['parameters_best_fit'].attrs[item] = str( self.best_fit[window][item] )
                
            # parameters
            OUT['parameters_bootstrap'] = xr.DataArray( np.nan, dims=('parameters', 'bootstrap'), coords={'parameters':pars, 'bootstrap':boot} )
            for par in pars:
                OUT['parameters_bootstrap'].loc[{'parameters':par}] = self.sol_bootstrap[window][par]
                
            # attribute
            OUT.attrs['name_data'] = self.name_data
            OUT.attrs['identifier_event'] = str(self.identifier_event)
            OUT.attrs['training_years'] = str(self.training_years)
            OUT.attrs['xtol_req'] = str(self.xtol_req)
            OUT.attrs['weighted_NLL'] = str(self.weighted_NLL)
            OUT.attrs['select_BIC_or_NLL'] = self.select_BIC_or_NLL

            # saving dataset
            OUT.to_netcdf( self.name_file_parameters(window) )#, encoding={var: {"zlib": True} for var in ['parameters_best_fit', 'parameters_bootstrap']} )
            
    def test_load_all( self ):
        return np.all( [os.path.isfile(self.name_file_parameters(window)) for window in self.list_windows] )
            
    def load_parameters(self):
        self.plot_trees = False
        self.best_fit = { window:{} for window in self.list_windows }
        self.sol_bootstrap = {window:{} for window in self.list_windows}

        for window in self.list_windows:
            # reading dataset
            ds = xr.open_dataset( self.name_file_parameters(window) )
            
            # best fit
            self.best_fit[window] = {'parameters':{par:ds['parameters_best_fit'].sel(parameters=par).values for par in ds.parameters.values} }
            self.best_fit[window]['distrib'] = ds['parameters_best_fit'].attrs['distrib']
            self.best_fit[window]['items_fit'] = eval( ds['parameters_best_fit'].attrs['items_fit'] )
            self.best_fit[window]['NLL'] = eval( ds['parameters_best_fit'].attrs['NLL'] )
            self.best_fit[window]['BIC'] = eval( ds['parameters_best_fit'].attrs['BIC'] )
            
            # parameters
            self.sol_bootstrap[window] = {par:ds['parameters_bootstrap'].sel(parameters=par).values for par in ds.parameters.values}
            
        # identifying everything that is missing
        self.n_iterations_BS = ds.bootstrap.size
        self.training_years = eval( ds.attrs['training_years'] )
        self.xtol_req = float( ds.attrs['xtol_req'] )
    #--------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------

