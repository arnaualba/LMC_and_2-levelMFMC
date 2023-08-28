import numpy as np
from sklearn.linear_model import lasso_path

class LassoOptim():
    '''
    It computes the full Lasso path on a given test set, using linear_model.lasso_path .
    Then it chooses the alpha that minimises the error on the test set.
    
    Parameters:
    Xtest : array of floats (N,d). The test inputs.
    ytest : array of floats (N). The test outputs.
    
    Arguments:
    Xtest_
    ytest_
    intercept_ : Intercept
    coef_ : Chosen best parameters
    alpha_ : Chosen best alpha
    alphas_ : array of alphas used in the lasso_path
    errors_train_ : errors along the path on y
    errors_ : errors along the path on ytest_
    
    Functions:
    fit(X,y,**params) : Computes the LassoPath with X,y, and the error of the path on y and ytest. 
                        **params are for the lasso_path function
    predict(X)
    '''
    def __init__(self, Xtest, ytest):
        super().__init__()
        self.Xtest_ = Xtest
        self.ytest_ = ytest.reshape(len(ytest),1)
        
    def fit(self, X, y, **params):
        self.intercept_ = y.mean()
        self.alphas_, coef_paths, _ = lasso_path(X, y - self.intercept_, **params)
        self.errors_train_ = np.power(np.matmul(X, coef_paths) + self.intercept_ - y.reshape(len(y),1),
                                      2).mean(axis = 0)
        self.errors_ = np.power(np.matmul(self.Xtest_, coef_paths) + self.intercept_ - self.ytest_, 
                                2).mean(axis = 0)
        idx = np.argmin(self.errors_)
        self.coef_ = coef_paths[:,idx]
        self.alpha_ = self.alphas_[idx]
    
    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_    