# Run these tests with python3 testsLMC.py to test any changes to the LMC class
# Probably need a conda environment for sklearn

import numpy as np
import time
from classLMC import LMC
from sklearn.linear_model import Lasso, LinearRegression, RidgeCV
import matplotlib.pyplot as plt

fs = 15  # Fontsize.
starttime = time.time()

# Function to check difference:
numberOfTests = 0
numberOfPassedTests = 0
def check(truth, state, tol = 1e-8):
    global numberOfTests
    global numberOfPassedTests
    numberOfTests += 1
    status = np.abs(truth - state) < tol
    if status.all() == True:
        numberOfPassedTests += 1
        print('Test passed')
    else:
        print('Test failed')
        print('Should be')
        print(truth)
        print('But is')
        print(state)

# Make test data:
# The true model is a linear function with added noise of std=eps, and dimension dim.
np.set_printoptions(formatter={'float': lambda x: format(x, '6.3E')})
eps = 0.1  
dim = 50
m = np.zeros(dim)  # Weights of the true model.
m[0] = 4.0
m[1] = 1.0
truemu = np.sum(m*0.5)  # Mean of uniform distribution is (0.5, 0.5, ..., 0.5)
truevar = 1/12*np.dot(m,m)  # Variance is m*Sigma*m'. Here Sigma=diag(1/12) since no correlations.
print("True mean is {:.2e}".format(truemu))
print("True var is {:.2e}".format(truevar))

# Test increasing dataset size.
lmc = LMC(regressor = Lasso(alpha = 0.002, max_iter = 10**4),
          random_state = np.random.randint(1000), verbose = False, 
          splitting_method = 'Nfold', use_alpha = False)
lmc_alpha = LMC(regressor = Lasso(alpha = 0.002, max_iter = 10**4),
                random_state = np.random.randint(1000), verbose = False, 
                splitting_method = 'Nfold', use_alpha = True)

Ns = np.logspace(1,3, num = 10)
print('Ns:', Ns)
Nrep = 15
meanMCs = np.zeros((Nrep, len(Ns)))
varMCs = np.zeros((Nrep, len(Ns)))
meanLMCs = np.zeros((Nrep, len(Ns)))
varLMCs = np.zeros((Nrep, len(Ns)))
meanLMCs_withAlpha = np.zeros((Nrep, len(Ns)))
varLMCs_withAlpha = np.zeros((Nrep, len(Ns)))
M = 10**5

for rep in range(Nrep):
    print()
    print("Repetition", rep)
    for i,N in enumerate(Ns):
        print("N =", N)
        Xtr = np.random.rand(int(np.floor(N)),dim)
        ytr = np.dot(Xtr,m) + eps * np.random.randn(Xtr.shape[0])  # Linear model with noise.
        Xte = np.random.rand(M,dim)

        results = lmc.get_estimates(Xtr,ytr,Xte)
        meanMCs[rep,i], varMCs[rep,i] = results['meanMC'], results['varMC']
        meanLMCs[rep,i], varLMCs[rep,i] = results['meanLMC'], results['varLMC']
        print("Regularisation parameter lambda:", results['regularisation_parameter'])
        print('numFeat is', len(np.where(results['regression_coefs'] != 0)[0]))
        
        results = lmc_alpha.get_estimates(Xtr,ytr,Xte)
        meanLMCs_withAlpha[rep,i], varLMCs_withAlpha[rep,i] = results['meanLMC'], results['varLMC']
        print('alphas: {:.2f}, {:.2f}'.format(results['alpha_mean'], results['alpha_var']))

fig1, axs = plt.subplots(1,2,figsize = (14,5))
axs = axs.reshape(-1)
fig1.subplots_adjust(wspace = .26)
axs[0].axhline(truemu, color = 'k', ls = '--', lw = 4)
x = Ns
for i in range(Nrep):
    axs[0].plot(x,meanMCs[i,:], color = 'blue', lw = 3)
    axs[0].plot(x,meanLMCs[i,:], color = 'darkorange', lw = 3)
    axs[0].plot(x,meanLMCs_withAlpha[i,:], color = 'green', lw = 3)
axs[0].set_ylabel('Estimation of mean', fontsize = fs)

axs[1].axhline(truevar, color = 'k', ls = '--', lw = 4)
for i in range(Nrep):
    axs[1].plot(x, varMCs[i,:], color = 'blue', lw = 3)
    axs[1].plot(x, varLMCs[i,:], color = 'darkorange', lw = 3, ls = '-')
    axs[1].plot(x, varLMCs_withAlpha[i,:], color = 'green', lw = 3)
axs[1].set_ylabel('Estimation of Variance', fontsize = fs)
axs[1].legend(['True Value', 'MC', 'LMC', 'LMC_alpha'], fontsize = fs)
for ax in axs:
    ax.set_xlabel('N', fontsize = fs)
    ax.tick_params( axis = 'both', labelsize = fs)
    ax.ticklabel_format( axis = 'both', style = 'sci', scilimits = (-1, 3))

fig2, axs = plt.subplots(1,2,figsize = (14,5))
axs = axs.reshape(-1)
fig2.subplots_adjust(wspace = .26)
axs[0].axhline(truemu, color = 'k', ls = '--', lw = 4)
x = Ns
y = meanMCs.mean(axis = 0)
yerr = meanMCs.std(axis = 0)
axs[0].fill_between(x, y-yerr, y+yerr, color = 'blue')
y = meanLMCs.mean(axis = 0)
yerr = meanLMCs.std(axis = 0)
axs[0].fill_between(x, y-yerr, y+yerr, color = 'darkorange',
                   hatch = '//', edgecolor = 'black')
y = meanLMCs_withAlpha.mean(axis = 0)
yerr = meanLMCs_withAlpha.std(axis = 0)
axs[0].fill_between(x, y-yerr, y+yerr, color = 'green',
                    hatch = '//', edgecolor = 'black')
axs[0].set_ylabel('Estimation of mean', fontsize = fs)
axs[1].axhline(truevar, color = 'k', ls = '--', lw = 4)
y = varMCs.mean(axis = 0)
yerr = varMCs.std(axis = 0)
axs[1].fill_between(x, y-yerr, y+yerr, color = 'blue')
y = varLMCs.mean(axis = 0)
yerr = varLMCs.std(axis = 0)
axs[1].fill_between(x, y-yerr, y+yerr, color = 'darkorange', 
                    hatch = '//', edgecolor = 'black')
y = varLMCs_withAlpha.mean(axis = 0)
yerr = varLMCs_withAlpha.std(axis = 0)
axs[1].fill_between(x, y-yerr, y+yerr, color = 'green', 
                    hatch = '//', edgecolor = 'black')
axs[1].set_ylabel('Estimation of Variance', fontsize = fs)
axs[1].legend(['True Value', 'MC', 'LMC', 'LMC_alpha'], fontsize = fs)
for ax in axs:
    ax.set_xlabel('N', fontsize = fs)
    ax.tick_params( axis = 'both', labelsize = fs)
    ax.ticklabel_format( axis = 'both', style = 'sci', scilimits = (-1, 3))

fig3, axs = plt.subplots(1,2,figsize = (14,5))
axs = axs.reshape(-1)
fig3.subplots_adjust(wspace = .26)
x = Ns
y = (np.abs(meanMCs - truemu) / truemu).mean(axis = 0)
axs[0].loglog(x, y, color = 'blue')
y = (np.abs(meanLMCs - truemu) / truemu).mean(axis = 0)
axs[0].loglog(x, y, color = 'darkorange')
y = (np.abs(meanLMCs_withAlpha - truemu) / truemu).mean(axis = 0)
axs[0].loglog(x, y, color = 'green')
axs[0].set_ylabel('Error in estimation of mean', fontsize = fs)
y = (np.abs(varMCs - truevar) / truevar).mean(axis = 0)
axs[1].loglog(x, y, color = 'blue')
y = (np.abs(varLMCs - truevar) / truevar).mean(axis = 0)
axs[1].loglog(x, y, color = 'darkorange')
y = (np.abs(varLMCs_withAlpha - truevar) / truevar).mean(axis = 0)
axs[1].loglog(x, y, color = 'green')
axs[1].set_ylabel('Error in estimation of Variance', fontsize = fs)
axs[1].legend(['MC', 'LMC', 'LMC_alpha'], fontsize = fs)
for ax in axs:
    ax.set_xlabel('N', fontsize = fs)
    ax.tick_params( axis = 'both', labelsize = fs)
    
plt.show()

# print("\nTest", numberOfTests,": Lasso regressor and bootstrapping for validation.")
# lmc = LassoMC(regressor = Lasso(alpha = 0.002, max_iter = 10**4),
#                inp_get = get_test_input, inp_out_get = get_test_input_and_output,
#                random_state = 42, desired_epsilon = 1e-4, verbose = False, 
#                NtestMax = 200, validation_method = 'bootstrap')
# lmc.MC_and_LMC(Nincr = 30, Nmin = 50, Nmax = 200,
#                checkLassoMC = True, checkMC = True)
# state = np.array([lmc.meanMCs[-1], lmc.stdMCs[-1],
#                   lmc.meanLMCs[-1], lmc.stdLMCs[-1]])
# truth = np.array([2.60799256, 1.14178364, 2.54192339, 1.21083688])
# check(truth, state)

# print("\nTest", numberOfTests,": Lasso regressor and split for validation.")
# lmc = LassoMC(regressor = Lasso(alpha = 0.002, max_iter = 10**4),
#                inp_get = get_test_input, inp_out_get = get_test_input_and_output,
#                random_state = 42, desired_epsilon = 1e-4, verbose = False, 
#                NtestMax = 200, validation_method = 'split')
# lmc.MC_and_LMC(Nincr = 30, Nmin = 50, Nmax = 200, checkLassoMC = True, checkMC = True)
# state = np.array([lmc.meanMCs[-1], lmc.stdMCs[-1],
#                   lmc.meanLMCs[-1], lmc.stdLMCs[-1]])
# truth = np.array([2.60799256, 1.14178364, 2.51052308, 1.21895404])
# check(truth, state)

# print("\nTest", numberOfTests,": Lasso regressor and 5Fold for validation.")
# lmc = LassoMC(regressor = Lasso(alpha = 0.002, max_iter = 10**4),
#                inp_get = get_test_input, inp_out_get = get_test_input_and_output,
#                random_state = 42, desired_epsilon = 1e-4, verbose = False, 
#                NtestMax = 200, validation_method = '5Fold')
# lmc.MC_and_LMC(Nincr = 30, Nmin = 50, Nmax = 200, Nwarmup = (10,40),
#                checkLassoMC = True, checkMC = True)
# state = np.array([lmc.meanMCs[-1], lmc.stdMCs[-1],
#                   lmc.meanLMCs[-1], lmc.stdLMCs[-1]])
# truth = np.array([2.59510571, 1.21880449, 2.52709387, 1.1607875 ])
# check(truth, state)

# print("\nTest", numberOfTests,": Lasso regressor and 10Fold for validation.")
# lmc = LassoMC(regressor = Lasso(alpha = 0.002, max_iter = 10**4),
#                inp_get = get_test_input, inp_out_get = get_test_input_and_output,
#                random_state = 42, desired_epsilon = 1e-4, verbose = False, 
#                NtestMax = 200,
#               validation_method = '5Fold', Nfold = 10)
# lmc.MC_and_LMC(Nincr = 30, Nmin = 50, Nmax = 200, Nwarmup = (8,40),
#                checkLassoMC = True, checkMC = True)
# state = np.array([lmc.meanMCs[-1], lmc.stdMCs[-1],
#                   lmc.meanLMCs[-1], lmc.stdLMCs[-1]])
# truth = np.array([2.55700622, 1.1950559, 2.44736305, 1.14048848])
# check(truth, state)

# print("\nTest", numberOfTests,": Lasso regressor and L2OCV for validation.",
#       "(LOOCV is not possible since you need at least 2 elements for var)")
# lmc = LassoMC(regressor = Lasso(alpha = 0.002, max_iter = 10**4),
#                inp_get = get_test_input, inp_out_get = get_test_input_and_output,
#                random_state = 42, desired_epsilon = 1e-4, verbose = False, 
#                NtestMax = 200,
#               validation_method = '5Fold', Nfold = 10000)  # Nfold>Nmax is equivalent to LOOCV.
# lmc.MC_and_LMC(Nincr = 30, Nmin = 50, Nmax = 200, Nwarmup = (8,40),
#                checkLassoMC = True, checkMC = True)
# state = np.array([lmc.meanMCs[-1], lmc.stdMCs[-1],
#                   lmc.meanLMCs[-1], lmc.stdLMCs[-1]])
# truth = np.array([2.60932477, 1.18498219, 2.46605861, 1.22393993])
# check(truth, state)

# print("\nTest", numberOfTests,": LassoCV for finding alpha, then Lasso for split.")
# lmc = LassoMC(inp_get = get_test_input, inp_out_get = get_test_input_and_output,
#                random_state = 42, desired_epsilon = 1e-4, verbose = False, 
#                NtestMax = 200, validation_method = 'split')
# lmc.MC_and_LMC(Nincr = 30, Nmin = 50, Nmax = 200, checkLassoMC = True, checkMC = True)
# state = np.array([lmc.meanMCs[-1], lmc.stdMCs[-1],
#                   lmc.meanLMCs[-1], lmc.stdLMCs[-1]])
# truth = np.array([2.60799256, 1.14178364, 2.51038628, 1.21814796])
# check(truth, state)

# print("\nTest", numberOfTests,": RidgeCV for finding alpha, then Ridge for split.")
# lmc = LassoMC(regressor = RidgeCV(cv = 5),
#               inp_get = get_test_input, inp_out_get = get_test_input_and_output,
#                random_state = 42, desired_epsilon = 1e-4, verbose = False, 
#                NtestMax = 200, validation_method = 'split')
# lmc.MC_and_LMC(Nincr = 30, Nmin = 50, Nmax = 200, Nwarmup = (10,40),
#                checkLassoMC = True, checkMC = True)
# state = np.array([lmc.meanMCs[-1], lmc.stdMCs[-1],
#                   lmc.meanLMCs[-1], lmc.stdLMCs[-1]])
# truth = np.array([2.40863723, 1.17155083, 2.59416893, 1.26091117])
# check(truth, state)

# print("\nTest", numberOfTests,": random ints")
# lmc = LassoMC(inp_get = get_test_input, inp_out_get = get_test_input_and_output,
#                random_state = 42, NtestMax = 10)
# state = np.array([lmc.__get_rs__(),lmc.__get_rs__()])
# truth = np.array([103, 436])
# check(truth, state)

# print("\nTest", numberOfTests,": get initial warmup samples")
# lmc = LassoMC(inp_get = get_test_input, inp_out_get = get_test_input_and_output,
#                random_state = 42, NtestMax = 10)
# lmc.__get_warmup_samples__(2,2)
# state = [lmc.Xtr, lmc.ytr, lmc.Xtest]
# truth = [
#     np.array([[0.75514015, 0.1430924, 0.8213222, 0.79774782, 0.26409003],
#               [0.98226464, 0.49234262, 0.2261136, 0.94855706, 0.63074856]]),
#     np.array([3.06457705, 4.51981948]),
#     np.array([[0.2370753, 0.74754928, 0.32864924, 0.11096434, 0.39369979],
#               [0.53766673, 0.64239968, 0.24518842, 0.62399067, 0.32235256]])
# ]
# for i in range(3):
#     check(truth[i], state[i])

# print("\nTest", numberOfTests,": get more samples")
# lmc = LassoMC(inp_get = get_test_input, inp_out_get = get_test_input_and_output,
#                random_state = 42, NtestMax = 10)
# lmc.__get_warmup_samples__(2,2)
# lmc.__get_more_samples__(Ntr = 2, Ntest = 2)
# lmc.__get_more_samples__(Ntr = 0, Ntest = 20)  # Add more than NtestMax
# state = [lmc.Xtr, lmc.ytr, lmc.Xtest]
# truth = [np.array([[0.75514015, 0.1430924 , 0.8213222 , 0.79774782, 0.26409003],
#                    [0.98226464, 0.49234262, 0.2261136 , 0.94855706, 0.63074856],
#                    [0.46003873, 0.89631548, 0.1901286 , 0.20620465, 0.51684131],
#                    [0.06925141, 0.75832032, 0.80444663, 0.89214894, 0.051145  ]]),
#          np.array([3.06457705, 4.51981948, 2.85974433, 0.88978694]),
#          np.array([[0.2370753 , 0.74754928, 0.32864924, 0.11096434, 0.39369979],
#                    [0.53766673, 0.64239968, 0.24518842, 0.62399067, 0.32235256],
#                    [0.63888475, 0.33764141, 0.08573119, 0.63663515, 0.44821628],
#                    [0.67648628, 0.28787749, 0.5424411 , 0.16631532, 0.49335388],
#                    [0.1067382 , 0.68434263, 0.53496262, 0.36918708, 0.41261464],
#                    [0.58784748, 0.71607379, 0.17333404, 0.06674229, 0.50067887],
#                    [0.4321472 , 0.94667592, 0.41625987, 0.24746969, 0.40414184],
#                    [0.63425908, 0.58376328, 0.78210436, 0.6168702 , 0.62776666],
#                    [0.36986912, 0.1683767 , 0.74532105, 0.39541609, 0.66053388],
#                    [0.31864425, 0.0694711 , 0.0489024 , 0.44164635, 0.70144592]])]
# for i in range(3):
#     check(truth[i], state[i])
# check(lmc.NtestMax, lmc.Xtest.shape[0])  # Check that Ntest doesn't become bigger that NtestMax

# print("\nTest", numberOfTests,": get_bootstrap_estimates")
# lmc = LassoMC(regressor = Lasso(alpha = 0.002, max_iter = 10**4),
#               random_state = 42, verbose = False, 
#               validation_method = '5Fold', Nfold = 10)
# Xtrain, ytrain = get_test_input_and_output(200, 42)
# ytest = get_test_input(200, 37)
# meanMC, stdMC, meanLMC, stdLMC = lmc.get_bootstrap_estimates(Xtrain, ytrain, ytest, Nrep = 5)
# state = np.array([meanMC, stdMC, meanLMC, stdLMC])
# truth = np.array(
#     [[2.40671302347374, 0.04726776884044894],
#      [1.205293545084826, 0.01766113926548377],
#      [2.5559885606124815, 0.004927978089134615],
#      [1.186464507273666, 0.005606011190437091]])
# check(truth, state)
    
# print('\n', numberOfPassedTests, '/', numberOfTests, 'tests were passed')
# totaltime = time.time() - starttime
# print("It took", totaltime, "seconds to run all the tests.")
# print("Bye, and have a nice day.")
