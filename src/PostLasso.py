class PostLasso():
    '''
    PostLasso regressor.
    The data is split into training and testing, and the Lasso path 
    is computed using linear_model.lasso_path, and evaluated on the test set.
    Then it chooses the optimal subset of features (support) and fits a linear model
    on the full data and the chosen support.
    
    Parameters:
    test_size : precentage of size to use for finding the optimal alpha.
    
    Arguments:
    alpha_ : Chosen best alpha
    alphas_ : array of alphas used in the lasso_path
    errors_train_ : errors along the path on y
    errors_ : errors along the path on ytest_
    support_ : indices of nonzero coefficients
    coef_ : coefficients from the Lasso fit (not from the linear model)
    reg_ : the LinearRegression regressor that is trained on sparse data
    
    Functions:
    fit(X,y,**params) : Computes the Lasso path and fits a linear model with the optimal features.
                        **params are for the lasso_path function
    predict(X)
    '''
    
    def __init__(self, test_size = 0.2):
        super().__init__()
        self.test_size_= test_size
        self.random_state_ = None
        self.alpha_ = None
        self.alphas_ = []
        self.support_ = []
        self.coef_ = []
        self.reg_ = LinearRegression()
        
    def fit(self, X, y, **params):
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, 
                                                        test_size = self.test_size_,
                                                        random_state = self.random_state_)
        self.alphas_, coef_paths, _ = lasso_path(Xtrain, ytrain, **params)
        self.errors_train_ = np.power(np.matmul(Xtrain, coef_paths) - ytrain.reshape(len(ytrain), 1),
                                      2).mean(axis = 0)
        self.errors_ = np.power(np.matmul(Xtest, coef_paths) - ytest.reshape(len(ytest), 1), 
                                2).mean(axis = 0)
        idx = np.argmin(self.errors_)
        self.alpha_ = self.alphas_[idx]
        self.support_ = np.where(coef_paths[:,idx] != 0.0)[0]
        self.coef_ = coef_paths[:,idx]
        if len(self.support_) == 0:
            self.support_ = [0] 
        self.reg_.fit(X[:,self.support_],y)
    
    def predict(self, X):
        return self.reg_.predict(X[:,self.support_])

    def get_params(self):
        params = {'numFeat' : len(self.support_)}                                                                                                                                                                      
        params['random_state'] = self.random_state_
        params['coef'] = self.coef_
        params['alphas'] = self.alphas_ 
        params['alpha'] = self.alpha_

        return params

    def set_params(self, **params):
        self.random_state_ = params['random_state']