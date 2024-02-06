import numpy as np

def get_power_sums(ys, max_order = 6):
    # Note the convention Ss[0] = sum(ys^1), Ss[1] = sum(ys^2), etc...
    Ss = np.zeros(max_order)
    for order in range(max_order):
        Ss[order] = np.sum(ys**(order+1))
    return Ss

def get_combined_power_sums(ys, ysSurr):
    # Note the convention SsComb[0,0] = N
    # SsComb[1,0] = sum((ys+ysSurr)), SsComb[0,1] = sum(ys-ysSurr)
    # SsComb[1,1] = sum((ys+ysSurr)(ys-ysSurr))
    # SsComb[1,2] = sum((ys+ysSurr)(ys-ysSurr)^2)
    # etc...
    SsComb = np.zeros((3,3))
    for i in range(2):
        for j in range(2):
            SsComb[i,j] =  np.sum((ys + ysSurr)**i * (ys - ysSurr)**j)
    return SsComb
    
def get_h_stats(Ss, N):
    hs = np.zeros(4)

    # Mean:
    hs[0] = Ss[0] / N
    # Variance:
    hs[1] = N*Ss[1]
    hs[1] += -Ss[0]**2
    hs[1] /= N * (N-1)
    # Skewness:
    hs[2] = N**2*Ss[2]
    hs[2] += -3*N*Ss[1]*Ss[0]
    hs[2] += 2*Ss[0]**3
    hs[2] /= N * (N-1) * (N-2)
    # Kurtosis:
    hs[3] = (-4*N**2 + 8*N - 12)*Ss[2]*Ss[0]
    hs[3] += (N**3 - 2*N**2 + 3*N)*Ss[3]
    hs[3] += 6*N*Ss[1]*Ss[0]**2
    hs[3] += (9 - 6*N)*Ss[1]**2
    hs[3] += -3*Ss[0]**4
    hs[3] /= N * (N-1) * (N-2) * (N-3)
    
    return hs

def get_Vars(Ss, N):
    # MC Vars.
    Vars = np.zeros(4)

    # Var mean:
    Vars[0] = N*Ss[1]
    Vars[0] += -Ss[0]**2
    Vars[0] /= N * (N-1)
    Vars[0] /= N
    # Var variance:
    Vars[1] = N*((N-1)**2*N*Ss[3] - (N**2-3)*Ss[1]**2)
    Vars[1] += (6-4*N)*Ss[0]**4
    Vars[1] += 4*N*(2*N-3)*Ss[1]*Ss[0]**2
    Vars[1] += -4*(N-1)**2*N*Ss[2]*Ss[0]
    Vars[1] /= N**2 * (N-1)**2 * (N-2) * (N-3)
    # Var skewness:
    # At the moment this is commented out, because it only
    # makes sense if combined_var skewness is available too.
    # term1 = 1 / ((N - 5)*(N - 4)*(N - 3)*(N - 2)**2*(N - 1)**2*N**2)
    # term2 = -12*(3*N**2 - 15*N + 20)*Ss[0]**6
    # term3 = 36*N*(3*N**2 - 15*N + 20)*Ss[1]*Ss[0]**4
    # term4 = -24*N**2*(2*N**2 - 9*N + 11)*Ss[2]*Ss[0]**3
    # term5 = 3*N*Ss[0]**2 * ((7*N**4 - 36*N**3 + 79*N**2 - 90*N + 40)*Ss[3]
    #                        - 6*N*(4*N**2 - 21*N + 29)*Ss[1]**2)
    # term6 = -6*N*Ss[0] * ((N**3 - 3*N**2 + 6*N - 8)*(N - 1)**2 * Ss[4]
    #                      + (-5*N**4 + 18*N**3 + 13*N**2 - 90*N + 40)*Ss[1]*Ss[2])
    # term7 = N * ((N - 1)**2 * N * (N**3 - 3*N**2 + 6*N - 8) * Ss[5] +
    #              3*(3*N**4 - 24*N**3 + 71*N**2 - 90*N + 40)*Ss[1]**3 -
    #              3*(2*N**5 - 11*N**4 + 14*N**3 + 25*N**2 - 70*N + 40)*Ss[3]*Ss[1] -
    #              (N**5 + 4*N**4 - 41*N**3 + 40*N**2 + 100*N - 80)*Ss[2]**2)
    # Vars[2] = term1 * (term2 + term3 + term4 + term5 - term6 + term7)

    # Var of higher orders not implemented yet.

    return Vars

def get_combined_Vars(Ss, N):
    # Here Ss is a matrix obtained with get_combined_power_sums().

    VarsComb = np.zeros(4)

    # MLMC Var mean, second part: Var(h1 - h1Surr) = Var(y - ySurr)/N
    VarsComb[0] = N*(Ss[0,2])
    VarsComb[0] += -Ss[0,1]**2
    VarsComb[0] /= N * (N-1)
    VarsComb[0] /= N
    # MLMC Var variance, second part: Var(h2 - h2Surr)
    VarsComb[1] = (-N**2 + N + 2) * Ss[1,1]**2
    VarsComb[1] += (N-1)**2 * (N*Ss[2,2] - 2*Ss[1,0]*Ss[1,2])
    VarsComb[1] += (N-1)*Ss[0,2]*(Ss[1,0]**2 - Ss[2,0])
    VarsComb[1] *= N
    VarsComb[1] += Ss[0,1]**2 * ((6-4*N)*Ss[1,0]**2 + (N-1)*N*Ss[2,0])
    VarsComb[1] += -2*N*Ss[0,1]*((N-1)**2*Ss[2,1] + (5-3*N)*Ss[1,0]*Ss[1,1])
    VarsComb[1] /= N**2 * (N-1)**2 * (N-2) * (N-3)
    # MLMC Var skewness, second part: Var(h3 - h3Surr)
    
    
    # Vars of higher orders not implemented yet.
    
    return VarsComb

def get_conf_ints(MSEs, p = 0.95):
    return np.array([np.sqrt(MSE / (1-p)) for MSE in MSEs])

def get_two_level_estimates(ys, ysSurr, ysSurrM,
                            calculate_MSEs = False,
                            adjust_alpha = False):
    """
    Returns a dictionary with the estimation of the first four moments
    using MC, the surrogate model surr, and two-level-MC.
    
    The moments are calculated using h-statistics.
    The high-fidelity model is considered unbiased and expensive,
    while the low-fidelity model surr is a cheap and biased surrogate model.
    
    ys and ysSurr are correlated and have N samples each.
    ysSurrM has M samples with M>>N, and is uncorrelated to ys and ysSurr.
    
    If calculate_MSEs, the unbiased MSEs of the moment estimations are estimated.
    They are returned as MSEs of MC and MSEs of MFMC.
    We have MSE_MFMC = Var(h(surr)) + Var(h(hi-fi) - h(surr)), so
    we also return the two parts of the MSE_MFMC as Vars surr and Vars comb.
    
    The two-level-MC formula for estimating a quantity m based on the h-statistics is
    m = h(ySurr, M) + h(y,N) - h(ySurr,N)
    If adjust_alpha = True, then the estimator becomes
    m = h(alpha*ySurr, M) + h(y,N) - h(alpha*ySurr,N),
    where alpha = cov(y,ySurr) / var(ySurr) is estimated from the available samples.
    """
    
    N = len(ys)
    M = len(ysSurrM)
    assert N == len(ysSurr), "There should be the same number of samples from ys and ysSurr."
    assert M > N, "There should be many more samples of ysSurrM than ys (M>>N)."

    # Compute alpha:
    if adjust_alpha:
        cov = np.cov(ysSurr, ys)
        alpha =  cov[0,1] / cov[0,0]  # cov(y, ySurr) / var(ySurr)
    else:
        alpha = 1.0

    # Get moments:
    Ss = get_power_sums(ys)
    hs = get_h_stats(Ss, N)
    SsSurr = get_power_sums(alpha * ysSurr)
    hsSurr = get_h_stats(SsSurr, N)
    SsSurrM = get_power_sums(alpha * ysSurrM)
    hsSurrM = get_h_stats(SsSurrM, M)

    return_dict = {'moments MC': hs,
                   'moments surr': hsSurrM,
                   'moments MFMC': hsSurrM + hs - hsSurr,
                   'adjust alpha' : adjust_alpha, 'alpha' : alpha}
    
    if calculate_MSEs:
        return_dict['MSEs MC'] = get_Vars(Ss, N)
        return_dict['Vars surr'] = get_Vars(SsSurrM, M)
        SsComb = get_combined_power_sums(ys, ysSurr)
        return_dict['Vars comb'] = get_combined_Vars(SsComb, N)
        return_dict['MSEs MFMC'] = return_dict['Vars surr'] + return_dict['Vars comb']
    
    return return_dict

    
