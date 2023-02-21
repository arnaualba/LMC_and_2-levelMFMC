import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LassoCV, LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import time


def dummy_get(*args, **kwargs):
    # This function does nothing.
    # If you use LMC only for "get_bootstrap_estimates",
    # then you don't need to define inp_get and inp_out_get, so you use this dummy function instead.
    color = 31  # Red.
    text = 'You have to define inp_get() and inp_out_get() in the initialization of the LMC class.'
    print("\033[1;" + str(color) + "m" + text + "\033[m")

def default_comp_mean(ytest, reg):
    return ytest.mean()

def default_comp_var(ytest, reg):
    return ytest.var(ddof=1)


class LassoMC():
    """
        Class for the LassoMC
    """
    def __init__(self, inp_get = dummy_get, inp_out_get = dummy_get,
                 regressor = LassoCV(cv = 5, max_iter = 10**4, eps = 1e-4, random_state = None, selection = 'random'),
                 random_state = None, desired_epsilon = 1e-2, verbose = 0,
                 NtestMax = 10**5, 
                 validation_method = 'bootstrap', Nfold = 5,
                 comp_mean = default_comp_mean, comp_var = default_comp_var):
        """
            inp_get : function to get an array of input vectors. Should be called as inp_get(N, random_state = None).
            inp_out_get : function to get an array of input vectors and an array of scalar outputs. Should be called as inp_out_get(N, random_state = None).
            regressor : sklearn regressor to use. By default it is a cross-validated Lasso regressor.
            random_state : random state to use for selecting samples and training. Default is None.
            desired_epsilon : repeat MC to converge ot a relative error desired_epsilon. Default value is 1e-2.
            verbose : 1 or 0.
            NtestMax : maximum number of samples Xtest. This limit is meant to avoid out-of-memory issues.
            validation_method : 'bootstrap' or 'split' or '5Fold'. If 'bootstrap', it makes a new set from Xtr by resampling, to compute the estimates.
                                With this method Xtr and Xval are huge, but overlap, so there is a risk of overfitting.
                                If 'split', Xtr will be split into trueXtr and Xval using 80/20%. 
                                This way Xval has never been seen for training, hence no overfitting. However Xtr and Xval are very small.
                                If '5Fold', 5 folds are used to train Xtr and validate on Xval. No overfitting, but very slow.
            Nfold : Number of folds. '5Fold' validation will default to 5 folds, but you can use more or less.
            comp_mean : function to compute mean of ytest. By default it's ytest.mean(), but one might prefer to use np.dot(reg.coef_, trueMean)
            comp_var : function to compute var of ytest. By default it's ytest.var(ddof=1), but one might prefer to use beta·trueSigma·beta
        """   
        self.inp_get = inp_get
        self.inp_out_get = inp_out_get
        self.reg = Pipeline([('scaler', StandardScaler()), ('regressor', regressor)])
        self.rs = np.random.RandomState(random_state)
        self.goalE = desired_epsilon
        self.goalE2 = desired_epsilon**2
        self.verbose = verbose
        self.NtestMax = NtestMax
        self.validation_method = validation_method
        self.Nfold = Nfold
        self.comp_mean = comp_mean
        self.comp_var = comp_var
        
        self.Ns = []
        self.meanMCs = []
        self.stdMCs = []
        self.meanLMCs = []
        self.stdLMCs = []
        self.timings = []
        
    def __get_rs__(self):
        return self.rs.randint(low = 1, high = 1000)
    
    def __get_warmup_samples__(self, Ntr, Ntest):
        self.Xtr, self.ytr = self.inp_out_get(Ntr, random_state = self.__get_rs__())
        self.Xtest = self.inp_get(Ntest, random_state = self.__get_rs__())

    def __get_more_samples__(self, Ntr, Ntest):
        Xtr, ytr = self.inp_out_get(Ntr, random_state = self.__get_rs__())
        self.Xtr = np.concatenate((self.Xtr, Xtr), axis = 0)
        self.ytr = np.concatenate((self.ytr, ytr), axis = 0)

        if Ntest + self.Xtest.shape[0] <= self.NtestMax:
            Xtest = self.inp_get(Ntest, random_state = self.__get_rs__())
            self.Xtest = np.concatenate((self.Xtest, Xtest), axis = 0)
        elif self.Xtest.shape[0] < self.NtestMax:
            Xtest = self.inp_get(self.NtestMax - self.Xtest.shape[0], random_state = self.__get_rs__())
            self.Xtest = np.concatenate((self.Xtest, Xtest), axis = 0)
        elif self.Xtest.shape[0] >= self.NtestMax and self.verbose:
            self.__print_warning__(
                "Xtest has reached the maximum size " +
                "NtestMax = {:.2e},".format(self.NtestMax) +
                " so it won't be increased further.")

    def __test_convergence__(self, checkMC = True, checkLassoMC = True):
        '''
        Tests whether algorithms have converged, by checking that the slope and std are small.
        The checkMC and checkLassoMC booleans say which algorithms to check for convergence.
        '''
        
        # Test that there are no trends in the last half of the problem.
        ## Get indexes of last half of the problem.
        N_2 = self.Ns[-1] / 2
        idx_2 = np.argmin(np.abs(np.array(self.Ns) - N_2))
        Ns = self.Ns[idx_2:]
        
        ## Get slope  of last half of the problem for MC estimates.
        meanMCs = self.meanMCs[idx_2:]
        slopeMuMC,_ = np.polyfit(Ns, meanMCs, 1)
        slopeMuMC /= np.mean(meanMCs)
        slopeMuMC = np.abs(slopeMuMC)
        
        stdMCs = self.stdMCs[idx_2:]
        slopeSigMC,_ = np.polyfit(Ns, stdMCs, 1)
        slopeSigMC /= np.mean(stdMCs)
        slopeSigMC = np.abs(slopeSigMC)

        ## Get slope  of last half of the problem for LassoMC estimates.
        meanLMCs = self.meanLMCs[idx_2:]
        slopeMuLMC,_ = np.polyfit(Ns, meanLMCs, 1)
        slopeMuLMC /= np.mean(meanLMCs)
        slopeMuLMC = np.abs(slopeMuLMC)
        
        stdLMCs = self.stdLMCs[idx_2:]
        slopeSigLMC,_ = np.polyfit(Ns, stdLMCs, 1)
        slopeSigLMC /= np.mean(stdLMCs)
        slopeSigLMC = np.abs(slopeSigLMC)

        # Now test that std is small in the last half of the problem.
        errMuMC = np.std(meanMCs) / np.mean(meanMCs)
        errSigMC = np.std(stdMCs) / np.mean(stdMCs)
        errMuLMC = np.std(meanLMCs) / np.mean(meanLMCs)
        errSigLMC = np.std(stdLMCs) / np.mean(stdLMCs)
        
        # Test if more sampled of Xtest are needed.
        Ntest_needed = (self.ytr**2).var() / self.goalE2
        Ntest = self.Xtest.shape[0]
        if Ntest_needed > Ntest:
            self.__get_more_samples__(0, int(np.ceil(Ntest_needed - Ntest)))

        if self.verbose:
            print('rel_slope mean MC: {:.2e}, LassoMC: {:.2e}'.format(slopeMuMC, slopeMuLMC),
                  '\nrel_slope std MC: {:.2e}, LassoMC: {:.2e}'.format(slopeSigMC, slopeSigLMC),
                  '\nrel error mean MC: {:.2e}, LassoMC: {:.2e}'.format(errMuMC, errMuLMC),
                  '\nrel error std MC: {:.2e}, LassoMC: {:.2e}'.format(errSigMC, errSigLMC))

        hasConvergedMC = ((slopeMuMC < self.goalE) and (slopeSigMC < self.goalE) and
                          (errMuMC < self.goalE) and (errSigMC < self.goalE))
        hasConvergedLassoMC = ((slopeMuLMC < self.goalE) and (slopeSigLMC < self.goalE) and
                               (errMuLMC < self.goalE) and (errSigLMC < self.goalE))

        return (hasConvergedMC or not checkMC) and (hasConvergedLassoMC or not checkLassoMC)
        
    def __train_regressor__(self):
        if 'random_state' in self.reg['regressor'].get_params().keys():
            self.reg['regressor'].set_params(**{'random_state' : self.__get_rs__()})

        # Only train a model if bootstrap,
        # or if regressor is a LassoCV or RidgeCV (and optimal alpha needs to be selected.
        # The 5Fold and Split methods train their own model later on.
        if (self.validation_method == 'bootstrap'
            or 'RidgeCV' in str(type(self.reg['regressor']))
            or 'LassoCV' in str(type(self.reg['regressor']))):
            mu = self.meanMCs[-1]
            sig = self.stdMCs[-1]
            self.reg.fit(self.Xtr, (self.ytr - mu) / sig)  # Fit with normalised output.

        if self.verbose:
            if 'alphas' in self.reg['regressor'].get_params().keys():
                print('optimal alpha is', self.reg['regressor'].alpha_)
            if 'numFeat' in self.reg['regressor'].get_params().keys():
                print('numFeat is', self.reg['regressor'].get_params()['numFeat'])
            if ('Lasso' in str(type(self.reg['regressor'])) and
                'coef_' in self.reg['regressor'].__dict__.keys()):
                print('numFeat is', len(np.where(self.reg['regressor'].coef_ != 0)[0]))
                
    def __2LMC__ (self, fullpred, valtrue, valpred, reg):
        mean = self.comp_mean(fullpred,reg) + (valtrue - valpred).mean()
        var = self.comp_var(fullpred,reg) + valtrue.var(ddof=1) - valpred.var(ddof=1)
        std = np.sqrt(var) if var > 0.0 else self.ytr.std(ddof=1)
        if var <= 0.0:
            self.__print_warning__(
                'Warning! Negative variance, so using simple MC.' +
                ' This usually happens when the Ntr is still small.'
            )
        return mean, std
    
    def __compute_estimates__(self):
        Ntr = len(self.ytr)
        self.Ns.append(Ntr)
        
        starttime = time.time()
        
        # MC estimates.
        self.meanMCs.append(self.ytr.mean())
        self.stdMCs.append(self.ytr.std(ddof=1))
        
        # LassoMC estimates.
        self.__train_regressor__()
        if (self.validation_method == '5Fold' or
            self.validation_method == 'split'):
            # In these cases, another new regressor regloc will be trained. If the self.reg regressor
            # is LassoCV or RidgeCV, then use simple Lasso/Ridge with the optimal alpha,
            # rather than redoing the costly CV method.
            if 'LassoCV' in str(type(self.reg['regressor'])):
                # We fit a Lasso model with the alpha chosen from our CV earlier.
                if self.verbose:
                    print('Using a normal Lasso model with the optimal alpha chosen by LassoCV')
                regloc = Pipeline([('scaler', self.reg['scaler']),
                                   ('regressor',
                                    Lasso(alpha = self.reg['regressor'].alpha_, max_iter = 10**4))])
            elif 'RidgeCV' in str(type(self.reg['regressor'])):
                # We fit a Ridge model with the alpha chosen from our CV earlier.
                if self.verbose:
                    print('Using a normal Ridge model with the optimal alpha chosen by RidgeCV')
                regloc = Pipeline([('scaler', self.reg['scaler']),
                                   ('regressor',
                                    Ridge(alpha = self.reg['regressor'].alpha_))])
            else:
                # If regressor was not a CV method, simply take the regressor as it is.
                regloc = self.reg
                
        # Now get the estimates:
        if self.validation_method == '5Fold':
            n_splits = self.Nfold if self.Nfold <= Ntr/2 else Ntr//2
            self.meanLMCs.append(0.0)
            self.stdLMCs.append(0.0)
            if self.verbose:
                print('Using ' + str(n_splits) + 'Fold estimation...')
            kf = KFold(n_splits = n_splits)
            for idxtr, idxval in kf.split(self.Xtr):
                Xtr, Xval = self.Xtr[idxtr], self.Xtr[idxval]
                ytr, self.yval = self.ytr[idxtr], self.ytr[idxval]
                mu = ytr.mean()
                sig = ytr.std()
                    
                if 'random_state' in regloc['regressor'].get_params().keys():
                    regloc['regressor'].set_params(**{'random_state' : self.__get_rs__()})
                    
                regloc.fit(Xtr, (ytr - mu) / sig)  # Fit with normalised output.
                
                self.yvalpred = regloc.predict(Xval) * sig + mu
                self.ytestpred = regloc.predict(self.Xtest) * sig + mu

                mean,std = self.__2LMC__(self.ytestpred, self.yval, self.yvalpred, regloc)
                self.meanLMCs[-1] += mean
                self.stdLMCs[-1] += std**2  # Compute mean of variances, and do sqrt in the end. The variance estimators are unbiased, not the std estimators.

            self.meanLMCs[-1] /= n_splits
            # self.stdLMCs[-1] /= n_splits
            self.stdLMCs[-1] = np.sqrt(self.stdLMCs[-1] / n_splits)

        elif self.validation_method == 'bootstrap':
            # Generate bootstrapped validation set.
            mu = self.meanMCs[-1]
            sig = self.stdMCs[-1]
            np.random.seed(self.__get_rs__())
            idxs = np.random.choice(Ntr, Ntr, replace = True)
            self.yval = self.ytr[idxs]
            self.yvalpred = self.reg.predict(self.Xtr[idxs]) * sig + mu

            # Get estimates.
            self.ytestpred = self.reg.predict(self.Xtest) * sig + mu
            mean, std = self.__2LMC__(self.ytestpred, self.yval, self.yvalpred, self.reg)
            self.meanLMCs.append(mean)
            self.stdLMCs.append(std)
            
        elif self.validation_method == 'split':
            Xtr, Xval, ytr, self.yval = train_test_split(self.Xtr, self.ytr,
                                                         test_size = 0.2,
                                                         random_state = self.__get_rs__())
            if self.verbose:
                print('Splitting Xtr into', Xtr.shape[0], 'training samples, and',
                      Xval.shape[0], 'validation samples.')
            mu = ytr.mean()
            sig = ytr.std(ddof = 1)
            regloc.fit(Xtr, (ytr - mu) / sig)  # Fit with normalised output.

            # Generate validation set.
            self.yvalpred = regloc.predict(Xval) * sig + mu
            self.ytestpred = regloc.predict(self.Xtest) * sig + mu
            mean,std = self.__2LMC__(self.ytestpred, self.yval, self.yvalpred, regloc)
            self.meanLMCs.append(mean)
            self.stdLMCs.append(std)
            
        else:
            print('Error, unknown validation method')
            return 1

        totaltime = time.time() - starttime
        if self.verbose:
            print('It took', totaltime, 'seconds to get the estimates.')
        self.timings.append(totaltime)
        
        if self.verbose and (
                self.validation_method == '5Fold' or
                self.validation_method == 'split'):
            if 'numFeat' in regloc['regressor'].get_params().keys():
                print('numFeat is', regloc['regressor'].get_params()['numFeat'])
            if ('Lasso' in str(type(regloc['regressor'])) and
                'coef_' in regloc['regressor'].__dict__.keys()):
                print('numFeat is', len(np.where(regloc['regressor'].coef_ != 0)[0]))

    def __print_num_samples__(self):
        if self.verbose:
            self.__print_warning__(
                'Ntr = ' + str(len(self.ytr)) + 
                ' labelled samples, Ntest = ' +
                str(self.Xtest.shape[0]) + 
                ' unlabelled samples.',
                color = 32)

    def __print_warning__(self, text, color = 33):
        # 30 = black, 31 = red, 32 = green, 33 = yellow, 34 = blue, 35 = magenta, 36 = cyan, 37 = white
        print("\033[1;" + str(color) + "m" + text + "\033[m")
    
    def MC_and_LMC(self, Nincr = 5, Nmin = 50, Nmax = 10**3, Nwarmup = (7,40),
                   checkMC = True, checkLassoMC = True):
        '''
        Method that does the MC and LMC convergence.
        At each pass of a loop it runs Nincr simulations (i.e. it gets Nincr (x,y) pairs).
        Then it computes the MC and LMC estimates.
        Once N > Nmin it starts checking if MC and LMC have converged.
        If N > Nmax it stops even if it hasn't converged.
        Nwarmup gives the initial number of samples for training (x,y pairs) and validation (just x)
        The checkMC and checkLassoMC booleans say which algorithms to check for convergence. This is in case you want to stop if just MC or just LMC has converged, but not the other.
        '''
        
        # Get initial samples.
        self.__get_warmup_samples__(Nwarmup[0], Nwarmup[1])
        self.__print_num_samples__()
        self.__compute_estimates__()

        # Increase labelled samples by Nincr until convergence of MC and LMC.
        conv = 0
        while not conv:
            self.__get_more_samples__(Nincr,0)
            self.__print_num_samples__()
            self.__compute_estimates__()
            # Check convergence only after Nmin samples have been obtained.
            if self.Ns[-1] > Nmin:
                conv = self.__test_convergence__(checkMC, checkLassoMC)
            # Quit if N becomes huge.
            if self.Ns[-1] + Nincr > Nmax:
                self.__print_warning__('\n\nQUITTING before convergence, because N > ' + str(Nmax))
                break

        # Print final state.
        self.__print_warning__("\nFinished with", color = 32)
        self.verbose = 1  # Such that final statement gets printed.
        self.__print_num_samples__()
        self.__test_convergence__()

    def get_bootstrap_estimates(self, Xtrain, ytrain, Xtest, Nrep = 5):
        Ntr = Xtrain.shape[0]
        self.Xtest = Xtest

        for i in range(Nrep):
            if self.verbose:
                print('\nBootstrap repetition', i+1)
                
            np.random.seed(self.__get_rs__())
            idxs = np.random.choice(Ntr, Ntr, replace = True)
            self.Xtr = Xtrain[idxs,:]
            self.ytr = ytrain[idxs]
            self.__print_num_samples__()
            self.__compute_estimates__()

            if self.verbose:
                print('MC estimates:', self.meanMCs[-1], self.stdMCs[-1])
                print('LassoMC estimates:', self.meanLMCs[-1], self.stdLMCs[-1])

        meanMC = [np.mean(self.meanMCs[-Nrep:]), np.std(self.meanMCs[-Nrep:])]
        stdMC = [np.mean(self.stdMCs[-Nrep:]), np.std(self.stdMCs[-Nrep:])]
        meanLMC = [np.mean(self.meanLMCs[-Nrep:]), np.std(self.meanLMCs[-Nrep:])]
        stdLMC = [np.mean(self.stdLMCs[-Nrep:]), np.std(self.stdLMCs[-Nrep:])]
                
        return meanMC, stdMC, meanLMC, stdLMC

    def get_single_estimate(self, Xtrain, ytrain, Xtest):
        Ntr = Xtrain.shape[0]
        self.Xtest = Xtest
        self.Xtr = Xtrain
        self.ytr = ytrain
        
        self.__print_num_samples__()
        self.__compute_estimates__()
        meanMC, stdMC = self.meanMCs[-1], self.stdMCs[-1]
        meanLMC, stdLMC = self.meanLMCs[-1], self.stdLMCs[-1]
        
        if self.verbose:
            print('MC estimates:', meanMC, "+-", stdMC)
            print('LMC estimates:', meanLMC, "+-", stdLMC)

        return meanMC, stdMC, meanLMC, stdLMC