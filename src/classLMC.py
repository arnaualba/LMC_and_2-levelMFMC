import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LassoCV, LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import time

def m4(x):
    '4th order central moment of the vector x'
    n = len(x)
    return np.sum((x - x.mean())**4) / (n-1)

def m22(x,y):
    'bivariate 2nd central moment'
    n = len(x)
    assert n == len(y)
    return np.sum((x - x.mean())**2 * (y - y.mean())**2) / (n-1)

class LMC():
    """
        Class for the Lasso Monte Carlo (LMC) method.
    
    """
    def __init__(self, regressor = LassoCV(cv = 5, max_iter = 10**4, eps = 1e-4, random_state = None, selection = 'random'),
                 random_state = None, verbose = 0,
                 splitting_method = 'Nfold',
                 Nfold = 5,
                 split_train_percent = 80,
                 use_alpha = True):
        """
            regressor : sklearn regressor to use. If not an sklearn class, it can be any regressor with 'fit', 'predict', and 'score' methods.
            random_state : positive int. Seed for reproducible results.
            verbose : 1 or 0.
            splitting_method :  'none', 'split' or 'Nfold'. Determines how Xtrain is used for training and estimating.
                                If 'none', Xtrain will be used for training and estimation.
                                This approach uses all the data, but risks overfitting on the training and introducing a bias in the estimation.
                                If 'split', Xtr will be split into a training set of split_train_percent, and a set of 100-split_train_percent for estimating.
                                This approach will have an unbiased estimation, but will have large variance due to the small training/estimation sets.
                                If 'Nfold', Nfold models will be trained, each with (N-1)/N portion of the data, and Nfold estimations will be made, each with 1/N portion of the data.
                                This approach is unbiased, and uses all the available data. 
            Nfold : Number of folds. Only used if splitting_method='Nfold'.
            split_train_percent : float in [0,100], or 'adaptive'. Percentage of data to use for training. Only used if splitting_method='split'
                                  If 'adaptive', it wi WIP
            use_alpha : boolean. If true, an alpha parameter is used as control variate coefficient.
        """   
        self.reg_ = Pipeline([('scaler', StandardScaler()), ('regressor', regressor)])
        self.rs_ = np.random.RandomState(random_state)
        self.verbose_ = verbose
        self.splitting_method_ = splitting_method
        self.Nfold_ = Nfold if self.splitting_method_ == 'Nfold' else None
        self.split_train_percent_ = split_train_percent if self.splitting_method_ == 'split' else None
        self.use_alpha_ = use_alpha

    def __get_rs__(self):
        'Returns a random int'
        return self.rs_.randint(low = 1, high = 10**5)
    
    def __print_warning__(self, text, color = 33):
        '''
        Prints the text in color. Default is 33 (yellow) for warnings. 
        30 = black, 31 = red (for errors), 32 = green, 33 = yellow, 34 = blue, 35 = magenta, 36 = cyan, 37 = white
        '''
        print("\033[1;" + str(color) + "m" + text + "\033[m")

    def __print_num_samples__(self):
        if self.verbose_:
            self.__print_warning__(
                'N = ' + str(len(self.ytr_)) + ' labelled samples, M = ' +
                str(self.Xte_.shape[0]) + ' unlabelled samples, and dim = ' +
                str(self.Xtr_.shape[1]) + ' input dimensions.',
                color = 32)

    def __2LMC__ (self, fullypred, ytrue, ypred, use_alpha = True):
        '''
        This function computes the 2-level estimators.
        It returns the estimated mean and variance,
        and also the arrays mean_errors and var_errors, which contain
        the estimation of the errors made in the estimation of the
        mean and variance.
        fullypred : array of size M. Contains many predictions of the surrogate model.
        ytrue : array of size N, with N<M. Contains the predictions of the expensive/true model.
        ypred : array of size N. Contains the predictions of the surrogate model.
        For ytrue and ypred, the same N input points were used. For fullypred, M input points were used,
        which are different and uncorrelated ot the N inputs used for ytrue and ypred.
        use_alpha : boolean. If true, an alpha parameter is used for the control variates.
        '''

        N = len(ypred)
        M = len(fullypred)

        varpred = np.var(ypred, ddof=1)
        vartrue = np.var(ytrue, ddof=1)
        m4pred = m4(ypred)

        # Compute optimal alpha coefficients for the control variates:
        if use_alpha:
            cov = np.cov(ypred, ytrue)[1,0]
            alpha_mean =  cov / varpred
            numerator = m22(ypred, ytrue) - varpred * vartrue + 2/(N+1)*cov**2
            alpha_var = np.sqrt(numerator /
                                (m4pred - varpred**2 + 2/(N+1)*varpred**2)) if numerator > 0 else 0.0
            if self.verbose_:
                print('Coefs for control variates: alpha_mean = ' +
                      '{:.2f}, and alpha_var = {:.2f}'.format(alpha_mean,
                                                              alpha_var))
        else:
            alpha_mean, alpha_var = 1.0, 1.0  # Equivalent to not choosing alphas.

        # Compute estimates:
        mean = alpha_mean * fullypred.mean() + (ytrue - alpha_mean * ypred).mean()
        var = alpha_var**2 * fullypred.var(ddof=1) + ytrue.var(ddof=1) - alpha_var**2 * ypred.var(ddof=1)

        # Estimate error in calculations of mean:
        mean_errors = np.zeros(2)
        mean_errors[0] = alpha_mean**2 * varpred / M  # First part of 2LMC error.
        mean_errors[1] = np.var(ytrue - alpha_mean*ypred, ddof=1) / N  # Second part of 2LMC error.

        # Estimate error in calculations of variance:
        var_errors = np.zeros(2)
        var_errors[0] = alpha_var**4 * (m4pred - (M-3)/(M-1)*varpred**2) / M  # First part of 2LMC error.
        var_errors[1] = (m22(ytrue + alpha_var*ypred, ytrue - alpha_var*ypred) +
                         1/(N-1)*np.var(ytrue + alpha_var*ypred, ddof=1)*np.var(ytrue - alpha_var*ypred, ddof=1) -
                         (N-2)/(N-1)*(vartrue - alpha_var**2*varpred)**2) / N  # Second part of 2LMC error.
        return mean, var, mean_errors, var_errors, alpha_mean, alpha_var

    def __train_regressor__(self, reg, Xtr, ytr):
        '''
        Trains the regressor on Xtr and ytr, and returns the regressor.
        reg : a pipeline that contains a 'scaler' and a 'regressor'
        Xtr, ytr : arrays containing the data to be trained on.
        '''
        
        if 'random_state' in reg['regressor'].get_params().keys():
            reg['regressor'].set_params(**{'random_state' : self.__get_rs__()})

        reg.fit(Xtr, ytr)

        if self.verbose_:
            if 'alphas' in reg['regressor'].get_params().keys():
                print('optimal regularisation parameter lambda is {:.2e}'.format(reg['regressor'].alpha_))
                
            if 'numFeat' in reg['regressor'].get_params().keys():
                print('numFeat is', reg['regressor'].get_params()['numFeat'])
            elif ('Lasso' in str(type(reg['regressor'])) and
                'coef_' in reg['regressor'].__dict__.keys()):
                print('numFeat is', len(np.where(reg['regressor'].coef_ != 0)[0]))

        return reg
    
    def get_estimates(self, Xtrain, ytrain, Xtest, ytest_lowlevel = [], ytrain_lowlevel = []):
        '''
        Compute LMC estimators.
        Inputs are all np.arrays.
        Xtrain : array of size (N,dim).
        ytrain : array of size (N). It contains the high-level predictions of Xtrain.
        Xtest : array of size (M,dim). It contains the unlabelled samples, with M >> N.
        If the previous arguments are provided, the LMC algorithm will be run. I.e. the necessary
        surrogate models will be trained, used to label Xtrain and Xtest, and compute the two-level
        estimators (2LMC).

        If both of the two following arguments are provided, then there is no need to train any surrogate
        model. Only the two-level MC estimators will be computed.
        ytest_lowlevel : array of size (M). Corresponds to low-level predictions of Xtest.
        ytrain_lowlevel : array of size (N). Corresponds to low-level predictions of Xtrain.
        '''
        self.Xte_ = Xtest
        self.Xtr_ = Xtrain
        self.ytr_ = ytrain
        N = len(self.ytr_)

        assert self.Xtr_.shape[1] == self.Xte_.shape[1]
        assert self.Xtr_.shape[0] == self.ytr_.shape[0]
        
        self.__print_num_samples__()
        
        starttime = time.time()
        
        # MC estimates:
        meanMC = self.ytr_.mean()
        varMC = self.ytr_.var(ddof=1)
        if varMC == 0:
            return meanMC, varMC, meanMC, varMC, np.zeros(2), np.zeros(2), np.zeros(2)

        m4MC = m4(self.ytr_)
        errorsMC = np.array([np.sqrt(varMC / N),
                             np.sqrt(m4MC - (N-3)/(N-1) * varMC**2) / np.sqrt(N)])

        # If low level predictions are provided, then we can do normal 2LMC, no need to train anything.
        if len(ytest_lowlevel) != 0 and len(ytrain_lowlevel) != 0:
            print('Computing normal 2LMC, since samples from the low level model were provided;')
            meanLMC, varLMC, MSE_LMC_mean, MSE_LMC_var, alpha_mean, alpha_var = self.__2LMC__(ytest_lowlevel,
                                                                                              self.ytr_, ytrain_lowlevel,
                                                                                              use_alpha = self.use_alpha_)
            # Prepare dictionary with main results:
            results = {'meanMC' : meanMC,
                       'varMC' : varMC,
                       'meanLMC' : meanLMC,
                       'varLMC' : varLMC,
                       'errorMC_mean' : errorsMC[0],
                       'errorMC_var' : errorsMC[1],
                       'MSE_LMC_mean' : MSE_LMC_mean,
                       'MSE_LMC_var' : MSE_LMC_var,
                       'alpha_mean' : alpha_mean,
                       'alpha_var' : alpha_var,
                       }

            if self.verbose_:
                totaltime = time.time() - starttime
                print('It took {:.2e} seconds to get the estimates.'.format(totaltime))
                self.__print_warning__('MC estimates: {:.3e}, {:.3e}'.format(meanMC, varMC))
                self.__print_warning__('LMC estimates: {:.3e}, {:.3e}'.format(meanLMC, varLMC))
                print('with estimated MC errors: {:.3e}, {:.3e}'.format(errorsMC[0], errorsMC[1]))
                errorsLMC = [np.sqrt(MSE_LMC_mean.sum()), np.sqrt(MSE_LMC_var.sum())]
                print('with estimated LMC errors: {:.3e}, {:.3e}'.format(errorsLMC[0], errorsLMC[1]))

            return results
        
        # Scale data, to make ML methods work better:
        self.ytr_ = (self.ytr_ - meanMC) / np.sqrt(varMC)
            
        # Select optimal regularisation parameter alpha (or lambda in the paper):
        if ((self.splitting_method_ == 'Nfold')
            and
            ('LassoCV' in str(type(self.reg_['regressor'])))):
            # In this case, Nfold models will be trained. If self.reg is LassoCV,
            # running the CV Nfold times will be very slow. Instead, we first train once to
            # find the optimal regularisation parameter alpha. Then, for the Nfold models
            # we will train normal Lasso models with the fixed alpha parameter.
            # In this way the expensive CV is only run once.
            self.reg_ = self.__train_regressor__(self.reg_, self.Xtr_, self.ytr_)
            regloc = Pipeline([('scaler', self.reg_['scaler']),
                               ('regressor',
                                Lasso(alpha = self.reg_['regressor'].alpha_, max_iter = 10**4))])
            if self.verbose_:
                print('Using a normal Lasso model with the' +
                      ' optimal lambda chosen by LassoCV')
        else:
            # If regressor was not a CV method, simply take the regressor as it is.
            regloc = self.reg_
                
        # Now get the estimates:
        if self.splitting_method_ == 'Nfold':
            n_split = self.Nfold_ if self.Nfold_ <= N/2 else N//2
            meanLMC = 0.0
            varLMC = 0.0
            MSE_LMC_mean = np.zeros(2)
            MSE_LMC_var = np.zeros(2)
            alpha_mean = 0.0
            alpha_var = 0.0
            if self.verbose_:
                print('Using ' + str(n_split) + 'Fold estimation...')
            kf = KFold(n_splits = n_split)
            for idxtr, idxpred in kf.split(self.Xtr_):
                Xtr, Xpred = self.Xtr_[idxtr], self.Xtr_[idxpred]
                ytr, ytrue = self.ytr_[idxtr], self.ytr_[idxpred]
                regloc = self.__train_regressor__(regloc, Xtr, ytr)
                
                ypred = regloc.predict(Xpred)
                fullypred = regloc.predict(self.Xte_)

                mean, var, mean_es, var_es, am, av = self.__2LMC__(fullypred,
                                                                   ytrue, ypred,
                                                                   use_alpha = self.use_alpha_)
                var = var if var > 0.0 else 1.0  # If statistics are poor, 2LMC could yield negative var.
                meanLMC += mean
                varLMC += var
                MSE_LMC_mean += mean_es
                MSE_LMC_var += var_es
                alpha_mean += am
                alpha_var += av

            meanLMC /= n_split
            varLMC /= n_split
            alpha_mean /= n_split
            alpha_var /= n_split

            # The first part of the LMC error is the average error of the Nfold trials.
            # Thus we divide by n_split.
            # However, the second part of the LMC error is the average error divided by the number of folds. (See LMC paper).
            MSE_LMC_mean[0] /= n_split
            MSE_LMC_var[0] /= n_split
            MSE_LMC_mean[1] /= n_split**2
            MSE_LMC_var[1] /= n_split**2
            

        elif self.splitting_method_ == 'none':
            # Get estimates.
            regloc = self.__train_regressor__(regloc, self.Xtr_, self.ytr_)
            
            fullypred = regloc.predict(self.Xte_)
            ypred = regloc.predict(self.Xtr_)
            meanLMC, varLMC, MSE_LMC_mean, MSE_LMC_var, alpha_mean, alpha_var = self.__2LMC__(fullypred,
                                                                                              self.ytr_, ypred,
                                                                                              use_alpha = self.use_alpha_)
            
        elif self.splitting_method_ == 'split':
            if (((type(self.split_train_percent_) is int) or (type(self.split_train_percent_) is float)) and
                (self.split_train_percent_ > 0) and (self.split_train_percent_ < 100)):
                Xtr, Xpred, ytr, ytrue = train_test_split(self.Xtr_, self.ytr_,
                                                          train_size = self.split_train_percent_ / 100,
                                                          random_state = self.__get_rs__())
                if self.verbose_:
                    print('Splitting Xtr into', Xtr.shape[0], 'training samples, and',
                          Xpred.shape[0], 'validation samples.')

            elif self.split_train_percent_ == 'adaptive':
                N = len(self.ytr_)
                if N < 10:
                    print("Since N < 10 the adaptive search cannot be done. Using 80% for training instead.")
                    percent_optim = 80
                else:
                    percents = np.arange(10,90,10)
                    MSEs = np.zeros(len(percents))
                    for i,percent in enumerate(percents):
                        Xtr, Xval, ytr, yval = train_test_split(self.Xtr_, self.ytr_,
                                                                  train_size = percent / 100,
                                                                  random_state = self.__get_rs__())
                        n = len(ytr)
                        regloc = self.__train_regressor__(regloc, Xtr, ytr)
                        yval_pred = regloc.predict(Xval)
                        MSEs[i] = np.var(yval_pred - yval) / (N - n)
                    percent_optim = percents[np.argmin(MSEs)]
                    print("Optimal chosen split is {:.2f}%, which corresponds to n={:d}".format(percent_optim, int(percent_optim*N/100)))
                Xtr, Xpred, ytr, ytrue = train_test_split(self.Xtr_, self.ytr_,
                                                          train_size = percent_optim / 100,
                                                          random_state = self.__get_rs__())
            else:
                print('Error, split_train_percent should be number in [0,100] or "adaptive"')
                return 1
            
            regloc = self.__train_regressor__(regloc, Xtr, ytr)
            fullypred = regloc.predict(self.Xte_)
            ypred = regloc.predict(Xpred)
            meanLMC, varLMC, MSE_LMC_mean, MSE_LMC_var, alpha_mean, alpha_var = self.__2LMC__(fullypred,
                                                                                              ytrue, ypred,
                                                                                              use_alpha = self.use_alpha_)
            
        else:
            print('Error, unknown splitting method.')
            return 1

        # Inversely scale the estimations.
        meanLMC = meanLMC * np.sqrt(varMC) + meanMC
        varLMC = varLMC * varMC if varLMC > 0.0 else varMC  # If statistics are poor, 2LMC could yield negative var.
        MSE_LMC_mean = varMC * MSE_LMC_mean  # Errors on mean.
        MSE_LMC_var= varMC**2 * MSE_LMC_var  # Errors on var.
        
        totaltime = time.time() - starttime
        if self.verbose_:
            print('It took {:.2e} seconds to get the estimates.'.format(totaltime))
            self.__print_warning__('MC estimates: {:.3e}, {:.3e}'.format(meanMC, varMC))
            self.__print_warning__('LMC estimates: {:.3e}, {:.3e}'.format(meanLMC, varLMC))
            print('with estimated MC errors: {:.3e}, {:.3e}'.format(errorsMC[0], errorsMC[1]))
            errorsLMC = [np.sqrt(MSE_LMC_mean.sum()), np.sqrt(MSE_LMC_var.sum())]
            print('with estimated LMC errors: {:.3e}, {:.3e}'.format(errorsLMC[0], errorsLMC[1]))
            error_ratios = [(errorsMC[0] - errorsLMC[0]) / errorsLMC[0], (errorsMC[1] - errorsLMC[1]) / errorsLMC[1]]
            threshold = 0.05  # How different two relative errors need to be, in order for one estimate to be considered better than the other.
            if error_ratios[0] > threshold and error_ratios[1] > threshold:
                print('The LMC errors are smaller, so LMC is probably more accurate.')
            elif error_ratios[0] < -threshold and error_ratios[1] < -threshold:
                print('The MC errors are smaller, so MC is probably more accurate.')                
            else:
                print("The errors are similar, so it's not clear whether MC or LMC is more accurate.")
            print('The two parts of the LMC mean MSE:', MSE_LMC_mean)
            if MSE_LMC_mean[0] > MSE_LMC_mean[1]:
                print('The first part of the LMC MSE is bigger. This scales as 1/M.')
            else:
                print('The second part of the LMC MSE is bigger. This scales as 1/N.')
            print('The two parts of the LMC var MSE:', MSE_LMC_var)
            if MSE_LMC_var[0] > MSE_LMC_var[1]:
                print('The first part of the LMC MSE is bigger. This scales as 1/M.')
            else:
                print('The second part of the LMC MSE is bigger. This scales as 1/N.')


        # Prepare dictionary with main results:
        results = {'meanMC' : meanMC,
                   'varMC' : varMC,
                   'meanLMC' : meanLMC,
                   'varLMC' : varLMC,
                   'errorMC_mean' : errorsMC[0],
                   'errorMC_var' : errorsMC[1],
                   'MSE_LMC_mean' : MSE_LMC_mean,
                   'MSE_LMC_var' : MSE_LMC_var,
                   }
        
        # Add other info to the dictionary:
        attrs = dir(regloc['regressor'])
        if 'alpha' in attrs:
            results['regularisation_parameter'] = regloc['regressor'].alpha  # This parameter is 'lambda' in the paper.
        elif 'alpha_' in attrs:
            results['regularisation_parameter'] = regloc['regressor'].alpha_
        if 'coef_' in attrs:
            results['regression_coefs'] = regloc['regressor'].coef_
        if self.use_alpha_:
            # Coefficients from control variates.
            results['alpha_mean'] = alpha_mean
            results['alpha_var'] = alpha_var
        
        return results

