import numpy as np
from sklearn.linear_model import lasso_path

class LassoFixedN():
    '''
    Lasso model where alpha is tuned such that the number of nonzero coefs is N.
    The Lasso path is computed on the training data, then alpha is chosen such that supp(coef_) = N,
    where supp is the support.
    Note: at the moment there are a bunch of internal reshape() operations. In the future this should be done in a nicer way.

    Parameters:
    Nfeat : float. If Nfeat > 1, number of nonzero coefficients.
    If Nfeat < 1.0, the number of nonzero coefficients is Nfeat*Ntrain
    output_scaler : None or a scaler from sklearn.preprocessing (e.g. StandardScaler or PowerTransformer, necessary since Lasso assumes a "linear model with Gaussian noise")
    y, since fit() and predict() will take care of it.
    
    Arguments:
    desiredNfeat_ : desired number of nonzero coefficients.
    output_scaler_ : None or a scaler from sklearn.preprocessing (e.g. StandardScaler or PowerTransformer, necessary since Lasso assumes a "linear model with Gaussian noise")
    Nfeat_ : true number of nonzero coefficients.
    alpha_ : Chosen best alpha
    alphas_ : array of alphas used in the lasso_path
    coef_ : coefficients from the Lasso fit.
    
    Functions:
    fit(X,y,**params) : Computes the LassoPath with X,y, and chooses the model with the desired number of features.
                        **params are for the lasso_path function
    predict(X)
    '''
    
    def __init__(self, Nfeat = 0.8, output_scaler = None):
        super().__init__()
        self.desiredNfeat_ = Nfeat
        self.output_scaler_ = output_scaler
        
        self.Nfeat_ = 0
        self.coef_ = []
        self.alphas_ = []
        self.alpha_ = 0.0
        
    def fit(self, X, y, **params):
        # Find target number of features.
        N = X.shape[0]
        targetN = int(self.desiredNfeat_) if self.desiredNfeat_ > 1.0 else int(self.desiredNfeat_ * N)

        # Preprocessing of output data:
        if self.output_scaler_ != None:
            y_transformed = y.reshape(-1,1)
            self.output_scaler_.fit(y_transformed)
            y_transformed = self.output_scaler_.transform(y_transformed).reshape(-1)
        else:
            y_transformed = y.copy()
            
        # Lasso path and find the path with the correct number of features.
        self.alphas_, coef_paths, _ = lasso_path(X, y_transformed, **params)
        
        Nalpha = len(self.alphas_)
        Nfeats = np.array([len(np.where(coef_paths[:,idx] != 0.0)[0]) for idx in range(Nalpha)])
        idx = np.argmin(np.abs(Nfeats - targetN))
        
        # Fix the final model.
        self.alpha_ = self.alphas_[idx]
        self.coef_ = coef_paths[:,idx]
        self.Nfeat_ = len(np.where(coef_paths[:,idx] != 0.0)[0])
    
    def predict(self, X):
        if self.output_scaler_ != None:
            return self.output_scaler_.inverse_transform(np.dot(X, self.coef_).reshape(-1,1)).reshape(-1)
        else:
            return np.dot(X, self.coef_)

    def get_params(self, deep = False):
        params = {'numFeat' : self.Nfeat_}   
        params = {'desiredNumFeat' : self.desiredNfeat_}
        params['coef'] = self.coef_
        params['alphas'] = self.alphas_ 
        params['alpha'] = self.alpha_
        
        return params
