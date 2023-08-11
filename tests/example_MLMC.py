# Run these tests with python3 testsLMC.py to test any changes to the LMC class
# Probably need a conda environment for sklearn

import numpy as np
import time
from classLMC import LMC
import matplotlib.pyplot as plt

fs = 15

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
lmc = LMC(random_state = np.random.randint(1000), verbose = False, use_alpha = False)
lmc_alpha = LMC(random_state = np.random.randint(1000), verbose = False, use_alpha = True)

Ns = np.logspace(1,3, num = 10)
print('Ns:', Ns)
Nrep = 15
meanMCs = np.zeros((Nrep, len(Ns)))
varMCs = np.zeros((Nrep, len(Ns)))
meanMLMCs = np.zeros((Nrep, len(Ns)))
varMLMCs = np.zeros((Nrep, len(Ns)))
meanMLMCs_withAlpha = np.zeros((Nrep, len(Ns)))
varMLMCs_withAlpha = np.zeros((Nrep, len(Ns)))
M = 10**4

for rep in range(Nrep):
    print()
    print("Repetition", rep)
    for i,N in enumerate(Ns):
        print("N =", N)
        Xtr = np.random.rand(int(np.floor(N)),dim)
        ytr = np.dot(Xtr,m) + eps * np.random.randn(Xtr.shape[0])  # Linear model with noise.
        Xte = np.random.rand(M,dim)
        yte = np.dot(Xte,m)  # The "low-level" model is a linear model without the noise.
        ytr_ll = np.dot(Xtr,m)

        results = lmc.get_estimates(Xtr,ytr,Xte, ytest_lowlevel = yte, ytrain_lowlevel = ytr_ll)
        meanMCs[rep,i], varMCs[rep,i] = results['meanMC'], results['varMC']
        meanMLMCs[rep,i], varMLMCs[rep,i] = results['meanLMC'], results['varLMC']
        
        results = lmc_alpha.get_estimates(Xtr,ytr,Xte, ytest_lowlevel = yte, ytrain_lowlevel = ytr_ll)
        meanMLMCs_withAlpha[rep,i], varMLMCs_withAlpha[rep,i] = results['meanLMC'], results['varLMC']
        print('alphas: {:.2f}, {:.2f}'.format(results['alpha_mean'], results['alpha_var']))

fig1, axs = plt.subplots(1,2,figsize = (14,5))
axs = axs.reshape(-1)
fig1.subplots_adjust(wspace = .26)
axs[0].axhline(truemu, color = 'k', ls = '--', lw = 4)
x = Ns
for i in range(Nrep):
    axs[0].plot(x,meanMCs[i,:], color = 'blue', lw = 3)
    axs[0].plot(x,meanMLMCs[i,:], color = 'darkorange', lw = 3)
    axs[0].plot(x,meanMLMCs_withAlpha[i,:], color = 'green', lw = 3)
axs[0].set_ylabel('Estimation of mean', fontsize = fs)

axs[1].axhline(truevar, color = 'k', ls = '--', lw = 4)
for i in range(Nrep):
    axs[1].plot(x, varMCs[i,:], color = 'blue', lw = 3)
    axs[1].plot(x, varMLMCs[i,:], color = 'darkorange', lw = 3, ls = '-')
    axs[1].plot(x, varMLMCs_withAlpha[i,:], color = 'green', lw = 3)
axs[1].set_ylabel('Estimation of Variance', fontsize = fs)
axs[1].legend(['True Value', 'MC', 'MLMC', 'MLMC_alpha'], fontsize = fs)
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
y = meanMLMCs.mean(axis = 0)
yerr = meanMLMCs.std(axis = 0)
axs[0].fill_between(x, y-yerr, y+yerr, color = 'darkorange',
                   hatch = '//', edgecolor = 'black')
y = meanMLMCs_withAlpha.mean(axis = 0)
yerr = meanMLMCs_withAlpha.std(axis = 0)
axs[0].fill_between(x, y-yerr, y+yerr, color = 'green',
                    hatch = '//', edgecolor = 'black')
axs[0].set_ylabel('Estimation of mean', fontsize = fs)
axs[1].axhline(truevar, color = 'k', ls = '--', lw = 4)
y = varMCs.mean(axis = 0)
yerr = varMCs.std(axis = 0)
axs[1].fill_between(x, y-yerr, y+yerr, color = 'blue')
y = varMLMCs.mean(axis = 0)
yerr = varMLMCs.std(axis = 0)
axs[1].fill_between(x, y-yerr, y+yerr, color = 'darkorange', 
                    hatch = '//', edgecolor = 'black')
y = varMLMCs_withAlpha.mean(axis = 0)
yerr = varMLMCs_withAlpha.std(axis = 0)
axs[1].fill_between(x, y-yerr, y+yerr, color = 'green', 
                    hatch = '//', edgecolor = 'black')
axs[1].set_ylabel('Estimation of Variance', fontsize = fs)
axs[1].legend(['True Value', 'MC', 'MLMC', 'MLMC_alpha'], fontsize = fs)
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
y = (np.abs(meanMLMCs - truemu) / truemu).mean(axis = 0)
axs[0].loglog(x, y, color = 'darkorange')
y = (np.abs(meanMLMCs_withAlpha - truemu) / truemu).mean(axis = 0)
axs[0].loglog(x, y, color = 'green')
axs[0].set_ylabel('Error in estimation of mean', fontsize = fs)
y = (np.abs(varMCs - truevar) / truevar).mean(axis = 0)
axs[1].loglog(x, y, color = 'blue')
y = (np.abs(varMLMCs - truevar) / truevar).mean(axis = 0)
axs[1].loglog(x, y, color = 'darkorange')
y = (np.abs(varMLMCs_withAlpha - truevar) / truevar).mean(axis = 0)
axs[1].loglog(x, y, color = 'green')
axs[1].set_ylabel('Error in estimation of Variance', fontsize = fs)
axs[1].legend(['MC', 'MLMC', 'MLMC_alpha'], fontsize = fs)
for ax in axs:
    ax.set_xlabel('N', fontsize = fs)
    ax.tick_params( axis = 'both', labelsize = fs)
    
plt.show()
