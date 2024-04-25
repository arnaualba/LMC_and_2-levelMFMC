Repository with class for the LMC method and a two-level version of MFMC.

# Multifidelity Monte Carlo

A two-level version of MFMC is implemented in `src/two_level_MC.py`. The formulas are taken from the paper on MLMC [Krumscheid et al. 2020](https://www.sciencedirect.com/science/article/pii/S0021999120302400), and the implementation is also heavily based on [Peherstorfer et al. 2016](https://doi.org/10.1137/15M1046472).

A paper of MFMC applied to UQ with spent nuclear fuel is under preparation.

## Usage
```

>>> # Generate data or import simulation results:
>>> M = 10**4  # Number of cheap simulations.
>>> N = 20  # Number of expensive simulations.
>>> d = 30  # Input dimension
>>> xs = np.random.randn(M, d) # Example: M input samples of dimension d.
>>> ys = np.array(list(map(hi_fi_model, xs[:N])))  # N simulations with expensive high-fidelity model.
>>> ysSurr = np.array(list(map(low_fi_model, xs)))  # M simulations with cheap low-fidelity model.
>>> print(ys.shape, ysSurr.shape)
(20,) (10000,)
>>>
>>> # Apply MFMC to get moment estimates:
>>> results = get_two_level_estimates(ys, ysSurr[:N], ysSurr, calculate_MSEs = True, adjust_alpha = True)
>>>
>>> # Computed moments:
>>> results['moments MC']  # Mean, variance, skewness, kurtosis
array([ -4.47172337,  24.05847351,  14.71051787, 943.96656693])
>>> results['moments MFMC']
array([-4.47758822e-01,  6.02057815e+01, -2.17309562e+01,  1.05442748e+04])
>>>
>>> # Estimated Mean Square Error of the estimations (not implemented for fourth moment yet):
>>> results['MSEs MC']
array([1.20292368e+00, 2.23019071e+01, 1.53800313e+03, 0.00000000e+00])
>>> results['MSEs MFMC']
array([4.84667872e-02, 4.91110539e+00, 3.80873430e+02, 0.00000000e+00])

```


# LassoMonteCarlo


The class `src/classLMC.py` can be used for LMC (Lasso + two-level MC), and also for a simple two-level MC.

See some examples in `src/example_MLMC.py` and `src/testsLMC.py`

## Theory

The LMC paper with all the theory can be found in `reports_and_papers/LMC_alba.pdf`.

### Control variate coefficient alpha
More recently, I added a "control variate coefficient alpha", as is commonly done with the control variate approach (see e.g. [Qian et al. 2018](https://epubs.siam.org/doi/pdf/10.1137/17M1151006) or simply [wikipedia](https://en.wikipedia.org/wiki/Control_variates)). The explanation of how this coefficient is implemented in this repository can be found in `reports_and_papers/LMC_extended.pdf`


## Usage

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
  - If 'none', Xtrain will be used for training and estimation.
        This approach uses all the data, but risks overfitting on the training and introducing a bias in the estimation.
  - If 'split', Xtr will be split into a training set of split_train_percent, and a set of 100-split_train_percent for estimating.
        This approach will have an unbiased estimation, but will have large variance due to the small training/estimation sets.
        This approach is a "normal" two-level MC estimator, with some data being used for training and some being used for estimations.
  - If 'Nfold', Nfold models will be trained, each with (N-1)/N portion of the data, and Nfold estimations will be made, each with 1/N portion of the data.
        This approach is unbiased, and uses all the available data. 
        This approach is in fact the "LMC" approach used in the paper, and it is the novel idea in the work.
- Nfold : Number of folds. Only used if splitting_method='Nfold'.
- split_train_percent : float in [0,100]. Percentage of data to use for training. Only used if splitting_method='split'
- use_alpha : boolean. If true, an alpha parameter is used as control variate coefficient.
