import matplotlib.pyplot as plt
import numpy as np

from numba import float64, int64, void, njit, generated_jit
import numba as nb

from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error

import time
import pandas as pd

# Compare implementations of coordinate descent.
# Loss = 1/2*||y-X·beta||_2^2 + lambda*||beta||_1
# All methods assume that X and beta are normalised with std=1 and mean=0

def numpy_coordinate_descent(X, y,
                             lamb, beta,
                             tol = 1e-4, maxIter = 1000):
    XY = np.dot(X.T, y)
    XX = np.dot(X.T, X)
    Xres = XY - np.dot(XX,beta)  # X*(y-X·beta), with res = (y-X·beta)
    
    for i in range(maxIter):
        oldBeta = beta.copy()
        for j in range(len(beta)):
            if beta[j] != 0.0:
                Xres += XX[:,j]*beta[j]
            beta[j] = np.sign(Xres[j]) * np.maximum(np.abs(Xres[j]) - lamb, 0.0) / XX[j,j]
            if beta[j] != 0.0:
                Xres -= XX[:,j]*beta[j]
        if np.max(np.abs(beta - oldBeta)) < tol * np.max(np.abs(beta)):
            break

@njit
def numba_coordinate_descent(X, y,
                             lamb, beta,
                             tol = 1e-4, maxIter = 1000):
    XY = np.dot(X.T, y)
    XX = np.dot(X.T, X)
    Xres = XY - np.dot(XX,beta)  # X·(y-X·beta), with res = (y-X·beta)
    
    for i in range(maxIter):
        oldBeta = beta.copy()
        for j in range(len(beta)):
            if beta[j] != 0.0:
                Xres += XX[:,j]*beta[j]
            beta[j] = np.sign(Xres[j]) * np.maximum(np.abs(Xres[j]) - lamb, 0.0) / XX[j,j]
            if beta[j] != 0.0:
                Xres -= XX[:,j]*beta[j]
        if np.max(np.abs(beta - oldBeta)) < tol * np.max(np.abs(beta)):
            break
# Run once to force Numba to compile.
beta = np.zeros(2)
dummy = numba_coordinate_descent(np.ones((3,2)), np.ones(3),
                                 0.1, beta,
                                 maxIter = 3)

#WIP Try cuda jit and also pytroch
@cuda.jit
def cuda_coordinate_descent(X, y,
                             lamb, beta,
                             tol = 1e-4, maxIter = 1000):
    XY = np.dot(X.T, y)
    XX = np.dot(X.T, X)
    Xres = XY - np.dot(XX,beta)  # X·(y-X·beta), with res = (y-X·beta)
    
    for i in range(maxIter):
        oldBeta = beta.copy()
        for j in range(len(beta)):
            if beta[j] != 0.0:
                Xres += XX[:,j]*beta[j]
            beta[j] = np.sign(Xres[j]) * np.maximum(np.abs(Xres[j]) - lamb, 0.0) / XX[j,j]
            if beta[j] != 0.0:
                Xres -= XX[:,j]*beta[j]
        if np.max(np.abs(beta - oldBeta)) < tol * np.max(np.abs(beta)):
            break
# Run once to force Numba to compile.
beta = np.zeros(2)
dummy = numba_coordinate_descent(np.ones((3,2)), np.ones(3),
                                 0.1, beta,
                                 maxIter = 3) 

def run_experiment(df, N, d, lamb, eps = 0.05):
    Xtest = np.random.randn(N,d)
    X     = np.random.randn(N,d)
    Xtest -= X.mean(axis=0); Xtest /= X.std(axis=0)
    X     -= X.mean(axis=0); X     /= X.std(axis=0)
    
    m = np.array([5,-6,3,0.1])
    m = np.concatenate((m, np.random.randn(d-len(m))))
    
    ytest = np.dot(Xtest,m) + np.random.randn(N) * eps
    y     = np.dot(X,m)     + np.random.randn(N) * eps
    ytest -= y.mean(); ytest /= y.std()
    y     -= y.mean(); y     /= y.std()
    
    # SKlearn
    reg = Lasso(fit_intercept = False, alpha = lamb / N)
    start = time.time()
    reg.fit(X,y)
    duration = time.time() - start
    beta = reg.coef_
    df.loc[:,'SK'] += [mean_squared_error(ytest, np.dot(Xtest, beta)),
                       duration]

    # Numpy
    beta = np.zeros(d)
    start = time.time()
    numpy_coordinate_descent(X, y, lamb, beta)
    duration = time.time() - start
    df.loc[:,'Numpy'] += [mean_squared_error(ytest, np.dot(Xtest, beta)),
                          duration]

    # Numba
    beta = np.zeros(d)
    start = time.time()
    numba_coordinate_descent(X, y, lamb, beta)
    duration = time.time() - start
    df.loc[:,'Numba'] += [mean_squared_error(ytest, np.dot(Xtest, beta)),
                          duration]
    


Nrep = 20
N = 50
d = 500
lamb = 1.0
print('N=', N, 'd=', d, 'lambda=', lamb)
df = pd.DataFrame(0.0,
                  columns = ['SK', 'Numpy', 'Numba'],
                  index = ['MSE', 'mean time'])
for rep in range(Nrep):
    if (rep%5 == 0) or (Nrep < 11):
        print('Repetition', rep+1, 'of', Nrep)
    run_experiment(df, N, d, lamb)
df /= N
pd.options.display.float_format = '{:5,.2e}'.format
print(df)


# print('SK took on average {:.2e} seconds'.format(meanSK / Nrep))
# print('Numba took on average {:.2e} seconds'.format(meanNumba / Nrep))
# print('SK MSE = {:.2e}'.format(mseSK / Nrep))
# print('Numba MSE = {:.2e}'.format(mseNumba / Nrep))

# print(regNumba.beta)
    
# plt.plot(Xtest[:,0], ytest, 'o', label = 'ytest')
# plt.scatter(Xtest[:,0], regSK.predict(Xtest),s=80, facecolors='none', 
#             edgecolors='orange', label = 'SK')
# # plt.scatter(Xtest[:,0], regNaive.predict(Xtest), marker='+', 
# #             color = 'red', label = 'Naive')
# plt.scatter(Xtest[:,0], regNumba.predict(Xtest), marker='+', color = 'red', label='Numba')
# plt.legend(fontsize = 12)
# plt.show()