# LassoMonteCarlo

Repository with class for the LMC method. 

The class `src/classLMC.py` can be used for LMC (Lasso + two-level MC), and also for a simple two-level MC.

See some examples in `src/example_MLMC.py` and `src/testsLMC.py`

# Theory

The LMC paper with all the theory can be found in `reports_and_papers/LMC_alba.pdf`.

## Control variate coefficient alpha
More recently, I added a "control variate coefficient alpha", as is commonly done with the control variate approach (see e.g. [Qian et al. 2018](https://epubs.siam.org/doi/pdf/10.1137/17M1151006) or simply [wikipedia](https://en.wikipedia.org/wiki/Control_variates)). The explanation of how this coefficient is implemented in this repository can be found in `reports_and_papers/LMC_extended.pdf`


# Usage

Example:
```
from classLMC import LMC

lmc = LMC(self, regressor = Lasso(alpha = 0.02),
        random_state = None, verbose = 0,
        splitting_method = 'Nfold',
        Nfold = 5,
        split_train_percent = 80,
        use_alpha = True):

results = lmc.get_estimates(Xtr,ytr,Xte)  # Xtr and ytr are the N labelled samples. Xte are the M the unlabelled samples.
```

with 
- regressor : sklearn regressor to use. If not an sklearn class, it can be any regressor with 'fit', 'predict', and 'score' methods.
- random_state : positive int. Seed for reproducible results.
- verbose : 1 or 0.
- splitting_method :  'none', 'split' or 'Nfold'. Determines how Xtrain is used for training and estimating.
        If 'none', Xtrain will be used for training and estimation.
        This approach uses all the data, but risks overfitting on the training and introducing a bias in the estimation.
        If 'split', Xtr will be split into a training set of split_train_percent, and a set of 100-split_train_percent for estimating.
        This approach will have an unbiased estimation, but will have large variance due to the small training/estimation sets.
        This approach is a "normal" two-level MC estimator, with some data being used for training and some being used for estimations.
        If 'Nfold', Nfold models will be trained, each with (N-1)/N portion of the data, and Nfold estimations will be made, each with 1/N portion of the data.
        This approach is unbiased, and uses all the available data. 
        This approach is in fact the "LMC" approach used in the paper, and it is the novel idea in the work.
- Nfold : Number of folds. Only used if splitting_method='Nfold'.
- split_train_percent : float in [0,100]. Percentage of data to use for training. Only used if splitting_method='split'
- use_alpha : boolean. If true, an alpha parameter is used as control variate coefficient.
