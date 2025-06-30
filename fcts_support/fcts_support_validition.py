import os
import xarray as xr
import numpy as np
import scipy.stats as ss
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# quick functions... working correctly, but not elegant



#---------------------------------------------------------------------------------------------------------------------------
# ANALYSIS OF PERFORMANCE
class analysis_distrib:
    def __init__(self, obs, params):
        if len(obs) > 1:
            raise Exception("should have only one window")
        window = list(obs.keys())[0]
        self.obs = obs[window]
        self.params = params[window]

        # BASIC APPROACH: transforming sample onto a GEV(0,0,shape)
        self.sample_transf = ((self.obs - self.params['loc']) / self.params['scale']).sel(label='with_CC')
        self.sample_cdf = self.cdf_from_sample(self.sample_transf.values, axis=self.sample_transf.dims.index('time'))
        self.confid_sample_cdf = np.percentile(self.sample_cdf[0], q=np.array([2.5,50,97.5]), axis=self.sample_transf.dims.index('bootstrap'))
        self.theoretical_quantiles = ss.genextreme.ppf(self.sample_cdf[1], c=-self.params['shape'].sel(label='with_CC', time=self.sample_transf.time))
        self.confid_theoretical_quantiles = np.percentile(self.theoretical_quantiles, q=np.array([2.5,50,97.5]), axis=0)
        # self.qq_plot()
        # counting where the sample is not within the range of the theoretical quantiles
        self.ind_median = np.argmin(np.abs(self.confid_theoretical_quantiles[1]))
        count_qq = np.where( (self.confid_theoretical_quantiles[0] > self.confid_sample_cdf[2])  |  (self.confid_theoretical_quantiles[2] < self.confid_sample_cdf[0]) )[0]
        count_qq_upper = count_qq[np.where( count_qq > self.ind_median )]
        self.frac_qq = len(count_qq) / len(self.confid_theoretical_quantiles[0])
        self.frac_qq_upper = len(count_qq_upper) / len(self.confid_theoretical_quantiles[0])
    
    def cdf_from_sample(self, data, axis):
        x, counts = np.unique(data, return_counts=True, axis=axis)
        cusum = np.cumsum(counts)
        return x, cusum / cusum[-1]

    def qq_plot(self, path_save):
        fig = plt.figure(figsize=(10,10))
        # id line
        plt.plot( self.confid_theoretical_quantiles[1], self.confid_theoretical_quantiles[1], lw=2, color='black', label="Identity line" )
        # range in theoretical quantile due to uncertainty in shape
        plt.fill_between( self.confid_theoretical_quantiles[1], self.confid_theoretical_quantiles[0], self.confid_theoretical_quantiles[2], color='grey', alpha=0.3, label="Range in theoretical quantiles due to uncertainty in shape" )
        # values from sample
        err = np.array([self.confid_sample_cdf[1]-self.confid_sample_cdf[0], self.confid_sample_cdf[2]-self.confid_sample_cdf[1]])
        plt.errorbar( x=self.confid_theoretical_quantiles[1], y=self.confid_sample_cdf[1], yerr=err, color='r', marker='o', ms=3, lw=0, elinewidth=1, label="Quantiles of sample, with range from uncertainty in location & scale")
        plt.xlabel('Theoretical quantile', size=14)
        plt.ylabel('Quantile of sample', size=14)
        plt.legend(loc=0, prop={'size':14})
        fig.savefig( os.path.join( path_save, 'qqplot.jpg' ), dpi=450 )
        fig.savefig( os.path.join( path_save, 'qqplot.pdf' ) )
        return fig
#---------------------------------------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------------------------------------
# GRANGER CAUSALITY
def nonlinear_granger_causality(treg, gmt, max_lag=5, n_permutations=100, random_state=0, verbose=False):
    # Build lagged matrices
    N = len(treg)
    X_treg = np.column_stack([np.roll(treg, i) for i in range(1, max_lag+1)])
    X_gmt = np.column_stack([np.roll(gmt, i) for i in range(1, max_lag+1)])
    y = treg
    # Remove first max_lag rows (contain rolled values from end)
    X_treg = X_treg[max_lag:]
    X_gmt = X_gmt[max_lag:]
    y = y[max_lag:]
    # Model 1: treg lags only
    model1 = RandomForestRegressor(random_state=random_state)
    model1.fit(X_treg, y)
    y_pred1 = model1.predict(X_treg)
    mse1 = mean_squared_error(y, y_pred1)
    # Model 2: treg lags + gmt lags
    X_full = np.hstack([X_treg, X_gmt])
    model2 = RandomForestRegressor(random_state=random_state)
    model2.fit(X_full, y)
    y_pred2 = model2.predict(X_full)
    mse2 = mean_squared_error(y, y_pred2)
    # Permutation test: shuffle gmt lags and recompute mse2
    mse2_perm = []
    rng = np.random.RandomState(random_state)
    for _ in range(n_permutations):
        X_gmt_perm = X_gmt.copy()
        rng.shuffle(X_gmt_perm)
        X_full_perm = np.hstack([X_treg, X_gmt_perm])
        model_perm = RandomForestRegressor(random_state=random_state)
        model_perm.fit(X_full_perm, y)
        y_pred_perm = model_perm.predict(X_full_perm)
        mse2_perm.append(mean_squared_error(y, y_pred_perm))
    mse2_perm = np.array(mse2_perm)
    # p-value: fraction of permuted MSEs lower than observed MSE2
    pval = np.mean(mse2_perm < mse2)
    return {'mse_treg': mse1, 'mse_treg_gmt': mse2, 'pval': pval}


def granger_causality_pr(gmt_in, treg_in, max_order=50, verbose_granger=False, nonlinear=False):
    # preparation of data
    if 'member' in gmt_in.dims:
        gmt = gmt_in.mean('member')
    else:
        gmt = gmt_in
    if len(treg_in.keys()) == 1:
        treg = list(treg_in.items())[0][1]
    else:
        treg = treg_in
    # start from date without nan
    tnan = gmt.time.where(np.isnan(gmt*treg), drop=True).values
    if len(tnan)==0:
        t0 = gmt.time.values[0] - 1
    else:
        t0 = tnan[-1]
    tt = np.array([t for t in gmt.time.values if (t in treg.time.values) and (t>t0)])
    gmt = gmt.sel(time=tt).values
    treg = treg.sel(time=tt).values

    if nonlinear:
        # Use nonlinear Granger causality test
        result = nonlinear_granger_causality(treg, gmt, max_lag=min(5, max_order), n_permutations=100, random_state=0, verbose=verbose_granger)
        if verbose_granger:
            print(f"Non-linear Granger-causality p-value: {result['pval']:.3g}")
        return {'nonlinear': True, **result}

    # Select optimal k_ar_diff for VECM using AIC
    data_for_coint = np.column_stack([treg, gmt])
    list_determ = ['ci', 'nc', 'co', 'colo', 'lo', 'cli', 'li']
    hard_limit = len(gmt)//3-2
    full_list_aic = []
    for determ in list_determ:
        list_k, list_aic = range(1, min(max_order, hard_limit)), []
        for k in list_k:
            vecm = VECM(data_for_coint, k_ar_diff=k, deterministic=determ).fit()
            list_aic.append(aic_vecm(determ, data_for_coint.shape[1], k, vecm.llf))
        full_list_aic.append(list_aic)
    full_list_aic = np.array(full_list_aic)
    
    # Select best k_ar_diff based on minimum AIC
    inds = np.where(full_list_aic==np.min(full_list_aic))
    best_determ, best_k_ar_diff = list_determ[inds[0][0]], list_k[inds[1][0]]
    
    # test for cointegration (Johansen test)
    johansen_test = coint_johansen(data_for_coint, det_order=0, k_ar_diff=best_k_ar_diff)
    cointegrated = johansen_test.lr1[0] > johansen_test.cvt[0, 1]

    if cointegrated:
        vecm = VECM(data_for_coint, k_ar_diff=best_k_ar_diff, deterministic=best_determ).fit()
        causality = vecm.test_granger_causality(caused=0, causing=[1]) # treg in 0, gmst in 1.
        pval = causality.pvalue
        if verbose_granger:
            print(f"VECM Granger-causality p-value: {pval:.3g} (k_ar_diff={best_k_ar_diff})")
        return { "cointegrated": True, "pval": pval}
    
    else:
        # Not cointegrated: difference the data and use standard VAR on differences
        gmt_stat = np.diff(gmt)
        treg_stat = np.diff(treg)
        data = np.column_stack([treg_stat, gmt_stat])
        # Restrict list_k to allowable lags
        hard_limit = len(gmt_stat)//3-2
        list_k_var = [k for k in list_k if k < hard_limit]
        # Select optimal lag for VAR using AIC
        list_aic_var = [VAR(data).fit(i).aic for i in list_k_var]
        best_order = list_k_var[np.argmin(list_aic_var)]
        # Fit VAR model and perform Granger causality test
        granger_results = grangercausalitytests(data, [best_order], verbose=False)
        pval = granger_results[best_order][0]['ssr_ftest'][1]
        if verbose_granger:
            print(f"Granger-causality p-value: {pval:.3g} (order={best_order})")
        return {"cointegrated": False, "pval": pval}


def aic_vecm(determ, n_vars, k, llf):
    # Set number of deterministic terms per equation based on 'determ' argument
    if determ in ['nc', 'co', 'colo', 'lo']:
        num_determ_terms = 0
    elif determ in ['ci', 'li']:
        num_determ_terms = 1
    elif determ == 'cli':
        num_determ_terms = 2
    else:
        raise ValueError(f"Unknown deterministic option: {determ}")
    n_params = n_vars * n_vars * k + n_vars * num_determ_terms
    return -2 * llf + 2 * n_params
#---------------------------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------------------------
# CLASS TO HANDLE RESULTS FOR VALIDITY
class func_save_validity:
    def __init__(self, datasets, path_save, identifier_event, option_train_wo_event, weighted_NLL, select_BIC_or_NLL, n_ESMs_CMIP6=None):
        self.datasets = datasets
        self.path_save = path_save
        self.identifier_event = identifier_event
        self.option_train_wo_event = option_train_wo_event
        self.weighted_NLL = weighted_NLL
        self.select_BIC_or_NLL = select_BIC_or_NLL
        self.n_ESMs_CMIP6 = n_ESMs_CMIP6

        # preparing name file
        self.name_file()
        
        # preparing outputs
        self.OUT = xr.Dataset()
        self.OUT['goodness'] = xr.DataArray(np.nan, coords={'dataset':self.datasets}, dims=('dataset',))
        self.OUT['granger'] = xr.DataArray(np.nan, coords={'dataset':self.datasets}, dims=('dataset',))

    def name_file(self):
        basis = 'validation_' + str(self.identifier_event)
        info_years = 'w' + self.option_train_wo_event*'o' + 'evt'
        info_training = self.weighted_NLL*'weighted' + 'NLL' + '-selected' + self.select_BIC_or_NLL
        if self.n_ESMs_CMIP6 is None:
            info_sel = 'all-ESMs'
        else:
            info_sel = str(self.n_ESMs_CMIP6)+'-ESMs'
        self.full_path_save = os.path.join(self.path_save, basis + '_' + info_years + '_' + info_training + '_' + info_sel + '.nc')
            
    def goodness(self, obs, evt_fits):
        self.perf_fit = {} # keeping this as variable to allow QQ-plot by users
        for name_data in self.datasets:
            self.perf_fit[name_data] = analysis_distrib(obs=obs[name_data], params=evt_fits[name_data].parameters)
            self.OUT['goodness'].loc[{'dataset':name_data}] = self.perf_fit[name_data].frac_qq_upper
        # check
        if self.OUT['goodness'].median() > 0.05:
            print("Will need to discard " + self.identifier_event + " due to insufficient goodness-of-fit")

    def granger(self, gmt_in, treg_in, nonlinear, verbose_granger=False):
        for name_data in self.datasets:
            test_granger = granger_causality_pr(gmt_in=gmt_in[name_data][1], treg_in=treg_in[name_data], nonlinear=nonlinear, verbose_granger=verbose_granger)
            self.OUT['granger'].loc[{'dataset':name_data}] = test_granger['pval']
        # check
        if self.OUT['granger'].median() > 0.05:
            print("Will need to discard " + self.identifier_event + " due to non-verified Granger causality")

    def test_load(self):
        return os.path.isfile(self.full_path_save)

    def save(self):
        self.OUT.to_netcdf(self.full_path_save)

    def load(self):
        self.OUT = xr.open_dataset(self.full_path_save)
        # check
        if self.OUT['goodness'].median() > 0.05:
            print("Will need to discard " + self.identifier_event + " due to insufficient goodness-of-fit")
        # check
        if self.OUT['granger'].median() > 0.05:
            print("Will need to discard " + self.identifier_event + " due to non-verified Granger causality")
#---------------------------------------------------------------------------------------------------------------------------



