import numpy as np
from sklearn.linear_model import lasso_path

class LassoFixedN():
    '''
    Lasso model where alpha is tuned such that the number of nonzero coefs is N.
    The Lasso path is computed on the training data, then alpha is chosen when
    
    Parameters:
    Nfeat : float. If Nfeat > 1, number of nonzero coefficients.
    If Nfeat < 1.0, the number of nonzero coefficients is Nfeat*Ntrain
    normalise_output : boolean. If True, there is no need to normalise the outputs
    y, since fit() and predict() will take care of it.
    
    Arguments:
    desiredNfeat_ : desired number of nonzero coefficients.
    normalise_output_ : whether to normalise the outputs y when training and predicting.
    Nfeat_ : true number of nonzero coefficients.
    alpha_ : Chosen best alpha
    alphas_ : array of alphas used in the lasso_path
    intercept_ : mean of training data.
    coef_ : coefficients from the Lasso fit.
    sig_ : std of training data.
    
    Functions:
    fit(X,y,**params) : Computes the LassoPath with X,y, and chooses the model with the desired number of features.
                        **params are for the lasso_path function
    predict(X)
    '''
    
    def __init__(self, Nfeat = 0.8, normalise_output = False):
        super().__init__()
        self.desiredNfeat_ = Nfeat
        self.normalise_output_ = normalise_output
        
        self.Nfeat_ = 0
        self.coef_ = []
        self.intercept_ = 0.0
        self.alphas_ = []
        self.alpha_ = 0.0
        
    def fit(self, X, y, **params):
        # Find target number of features.
        N = X.shape[0]
        targetN = int(self.desiredNfeat_) if self.desiredNfeat_ > 1.0 else int(self.desiredNfeat_ * N)
        
        # Lasso path and find the path with the correct number of features.
        self.intercept_ = y.mean()
        self.sig_ = y.std(ddof=1) if self.normalise_output_ else 1.0
        self.alphas_, coef_paths, _ = lasso_path(X,
                                                 (y - self.intercept_) / self.sig_,
                                                 **params)
        Nalpha = len(self.alphas_)
        Nfeats = np.array([len(np.where(coef_paths[:,idx] != 0.0)[0]) for idx in range(Nalpha)])
        idx = np.argmin(np.abs(Nfeats - targetN))
        
        # Fix the final model.
        self.alpha_ = self.alphas_[idx]
        self.coef_ = coef_paths[:,idx]
        self.Nfeat_ = len(np.where(coef_paths[:,idx] != 0.0)[0])
    
    def predict(self, X):
        return np.dot(X, self.coef_) * self.sig_ + self.intercept_

    def get_params(self):
        params = {'numFeat' : self.Nfeat_}   
        params = {'desiredNumFeat' : self.desiredNfeat_}
        params['coef'] = self.coef_
        params['intercept'] = self.intercept_
        params['sig'] = self.sig_
        params['alphas'] = self.alphas_ 
        params['alpha'] = self.alpha_
        
        return params
