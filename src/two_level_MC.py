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
    SsComb = np.zeros((7,7))
    for i in range(7):
        for j in range(7):
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
    Vars[2] = exaqute_xmc_variance_skewness(Ss, N) / N
    # Var Kurtosis not implemented.

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
    VarsComb[2] = exaqute_xmc_combined_variance_skewness(Ss, N) / N
    
    # Var Kurtosis not implemented.
    
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
        SsComb = get_combined_power_sums(ys, alpha * ysSurr)
        return_dict['Vars comb'] = get_combined_Vars(SsComb, N)
        return_dict['MSEs MFMC'] = return_dict['Vars surr'] + return_dict['Vars comb']
    
    return return_dict


def exaqute_xmc_variance_skewness(Ss, N):
    # Arnau: I have copied this formula from exaquate-XMC, since it is such a lengthy expression:
    # See source of paper and code:
    # Source: MATHICSE technical report 23.2017, table 1, p. 7
    t1 = N ** 2
    t2 = t1 * N
    t3 = t1 ** 2
    t6 = t3 * t1
    t13 = Ss[2] ** 2
    t15 = t3 * N
    t16 = Ss[0] ** 2
    t20 = t15 * Ss[0]
    t21 = Ss[1] * Ss[2]
    t24 = Ss[1] ** 2
    t25 = t24 * Ss[1]
    t28 = t16 * Ss[0]
    t32 = t3 * t16
    t35 = t16 ** 2
    t39 = t35 * t16
    t44 = (
        21 * t15 * t16 * Ss[3]
        + t3 * t2 * Ss[5]
        + 108 * t2 * t35 * Ss[1]
        - 48 * t3 * t28 * Ss[2]
        - 6 * t6 * Ss[0] * Ss[4]
        - 6 * t6 * Ss[1] * Ss[3]
        - 36 * t1 * t39
        - t6 * t13
        + 9 * t15 * t25
        + 30 * t20 * t21
        - 72 * t32 * t24
        - 5 * t6 * Ss[5]
    )
    t54 = t3 * Ss[0]
    t62 = t2 * t16
    t77 = (
        -540 * t1 * t35 * Ss[1]
        + 33 * t15 * Ss[1] * Ss[3]
        + 216 * t2 * t28 * Ss[2]
        - 42 * t3 * Ss[1] * Ss[3]
        + 180 * N * t39
        - 4 * t15 * t13
        + 13 * t15 * Ss[5]
        + 30 * t20 * Ss[4]
        - 108 * t54 * t21
        + 378 * t62 * t24
        - 72 * t3 * t25
        - 108 * t32 * Ss[3]
        - 78 * t54 * Ss[4]
    )
    t83 = t2 * Ss[0]
    t91 = t1 * t16
    t109 = (
        720 * N * t35 * Ss[1]
        - 264 * t1 * t28 * Ss[2]
        - 75 * t2 * Ss[1] * Ss[3]
        - 40 * t2 * t13
        + 41 * t3 * t13
        + 213 * t2 * t25
        - 78 * t83 * t21
        - 522 * t91 * t24
        - 23 * t3 * Ss[5]
        + 237 * t62 * Ss[3]
        + 138 * t83 * Ss[4]
        - 270 * t91 * Ss[3]
        - 240 * t39
    )
    t110 = t1 * Ss[0]
    t127 = N * Ss[0]
    t141 = (
        120 * N * t16 * Ss[3]
        - 120 * N * Ss[1] * Ss[3]
        + 210 * t1 * Ss[1] * Ss[3]
        + 80 * N * t13
        + 120 * N * t25
        - 100 * t1 * t13
        - 270 * t1 * t25
        - 8 * t1 * Ss[5]
        + 540 * t110 * t21
        - 132 * t110 * Ss[4]
        - 240 * t127 * t21
        + 48 * t127 * Ss[4]
        + 22 * t2 * Ss[5]
    )
    t155 = (-2 + N) ** 2
    t158 = (-1 + N) ** 2
    uglyD0O3E = (
        (t44 + t77 + t109 + t141)
        / N
        / (-5 + N)
        / (-4 + N)
        / (-3 + N)
        / t155
        / t158
    )
    return uglyD0O3E

def exaqute_xmc_combined_variance_skewness(Ss, N):
    # Arnau: I have copied this formula from exaquate-XMC, since it is such a lengthy expression:
    # See source of paper and code:
    # Source: MATHICSE technical report 23.2017, appendix B, p. 28
    t1 = N ** 2
    t2 = 3.0 * t1
    t4 = t2 - 15 * N + 20.0
    t5 = Ss[0,1] ** 2
    t6 = t5 ** 2
    t12 = Ss[1,0] ** 2
    t15 = -3.0 + N
    t16 = t15 ** 2
    t27 = 2.0 * t1 - 9.0 * N + 11.0
    t33 = N * t1
    t34 = 5.0 * t1
    t35 = 6.0 * N
    t36 = t33 - t34 + t35 + 2.0
    t46 = Ss[0,2] ** 2
    t49 = t1 ** 2
    t53 = 90.0 * N
    t58 = 8.0 * t1
    t62 = 3.0 * t33
    t64 = 30.0 * N
    t70 = t12 ** 2
    t79 = -2.0 + N
    t80 = -1.0 + N
    t81 = t80 ** 2
    t82 = t79 * t81
    t83 = N * Ss[1,0]
    t89 = Ss[1,1] ** 2
    t92 = 2.0 * t33
    t94 = 7.0 * N
    t96 = Ss[2,0] ** 2
    t100 = t1 - 3.0 * N + 2.0
    t115 = N * t49
    t122 = Ss[1,0] * t12
    t144 = (
        88.0 * N * Ss[2,3]
        + 66.0 * N * Ss[4,1]
        - 92.0 * t1 * Ss[2,3]
        + 3.0 * t115 * Ss[4,1]
        + 24.0 * Ss[0,3] * t12
        + 48.0 * t12 * Ss[2,1]
        - 696.0 * t122 * Ss[1,1]
        + 52.0 * t33 * Ss[2,3]
        + 39.0 * t33 * Ss[4,1]
        + 16.0 * Ss[0,3] * Ss[2,0]
        - 64.0 * Ss[1,0] * Ss[1,3]
        - 96.0 * Ss[1,0] * Ss[3,1]
        + 96.0 * Ss[1,1] * Ss[1,2]
        - 32.0 * Ss[2,3]
    )
    t165 = 3.0 * t49
    t174 = t49 * Ss[1,0]
    t177 = t33 * Ss[1,0]
    t180 = t1 * Ss[1,0]
    t185 = t49 * Ss[1,1]
    t188 = t1 * Ss[1,1]
    t191 = t33 * Ss[1,1]
    t194 = (
        -69.0 * t1 * Ss[4,1]
        + 72.0 * Ss[2,0] * Ss[2,1]
        - 20.0 * t49 * Ss[2,3]
        + 4.0 * t115 * Ss[2,3]
        + 48.0 * Ss[1,1] * Ss[3,0]
        - 15.0 * t49 * Ss[4,1]
        + Ss[0,2]
        * (
            (-5.0 * t49 + 18.0 * t33 + 13.0 * t1 - t53 + 40.0) * Ss[0,3]
            + 36.0 * t36 * Ss[1,0] * Ss[1,1]
            - 3.0 * (t165 - 14.0 * t33 + t1 + 50.0 * N - 16.0) * Ss[2,1]
        )
        - 24.0 * Ss[4,1]
        - 18.0 * t174 * Ss[3,1]
        + 96.0 * t177 * Ss[3,1]
        - 210.0 * t180 * Ss[3,1]
        + 228.0 * t83 * Ss[3,1]
        - 6.0 * t185 * Ss[3,0]
        - 6.0 * t188 * Ss[3,0]
        + 24.0 * t191 * Ss[3,0]
    )
    t196 = N * Ss[1,1]
    t201 = t1 * t122
    t207 = t33 * Ss[0,3]
    t210 = N * Ss[0,3]
    t217 = N * t122
    t220 = t1 * Ss[0,3]
    t231 = (
        -60.0 * t196 * Ss[3,0]
        + 90.0 * t188 * Ss[1,2]
        - 96.0 * t201 * Ss[1,1]
        + t81 * (t33 - t2 + t35 - 8.0) * Ss[0,5]
        + 12.0 * t207 * t12
        + 72.0 * t210 * t12
        - 180.0 * t196 * Ss[1,2]
        - 6.0 * t185 * Ss[1,2]
        + 504.0 * t217 * Ss[1,1]
        - 60.0 * t220 * t12
        + 132 * t83 * Ss[1,3]
        - 106.0 * t180 * Ss[1,3]
        - 10.0 * t174 * Ss[1,3]
        + 48.0 * t177 * Ss[1,3]
    )
    t235 = t49 * Ss[2,0]
    t238 = t33 * t12
    t241 = Ss[1,1] * Ss[2,0]
    t246 = t33 * Ss[2,0]
    t249 = t1 * t12
    t256 = t1 * Ss[2,0]
    t259 = N * t12
    t266 = N * Ss[2,0]
    t269 = Ss[1,0] * Ss[1,1]
    t272 = (
        -5.0 * t49 * Ss[0,3] * Ss[2,0]
        + 60.0 * t177 * t241
        - 324.0 * t180 * t241
        + 30.0 * t207 * Ss[2,0]
        - 30.0 * t210 * Ss[2,0]
        - 35.0 * t220 * Ss[2,0]
        - 9.0 * t235 * Ss[2,1]
        + 48.0 * t238 * Ss[2,1]
        + 480.0 * t83 * t241
        + 30.0 * t246 * Ss[2,1]
        - 228.0 * t249 * Ss[2,1]
        + 45.0 * t256 * Ss[2,1]
        + 276.0 * t259 * Ss[2,1]
        - 210.0 * t266 * Ss[2,1]
        - 72.0 * t269 * Ss[2,0]
    )
    t277 = Ss[2,1] ** 2
    t279 = Ss[1,2] ** 2
    t286 = Ss[0,3] ** 2
    t323 = (
        432.0 * t277
        + 288.0 * t279
        - (t115 + 4.0 * t49 - 41.0 * t33 + 40.0 * t1 + 100.0 * N - 80.0) * t286
        - 8.0 * N * Ss[0,6]
        + 22.0 * t1 * Ss[0,6]
        - 23.0 * t33 * Ss[0,6]
        + 864.0 * t89 * Ss[2,0]
        - 138.0 * t33 * Ss[2,4]
        - 432.0 * Ss[2,0] * Ss[2,2]
        + 6.0
        * Ss[0,3]
        * (
            4.0 * (t49 - t62 - t58 + t64 - 8.0) * Ss[1,0] * Ss[1,1]
            - (t115 - 2.0 * t49 - 17.0 * t33 + 34.0 * t1 + 40.0 * N - 32.0)
            * Ss[2,1]
        )
        + 3.0 * (t165 - 24.0 * t33 + 71.0 * t1 - t53 + 40.0) * Ss[0,2] * t46
        - 540.0 * N * t277
        - 48.0 * Ss[0,4] * Ss[2,0]
    )
    t324 = t1 * t49
    t351 = (
        -72.0 * N * Ss[4,2]
        + 132.0 * t1 * Ss[2,4]
        - 30.0 * t115 * Ss[2,4]
        - 45.0 * t115 * Ss[4,2]
        - 144.0 * t12 * t89
        + 432.0 * t12 * Ss[2,2]
        + 144.0 * t122 * Ss[1,2]
        + 6.0 * t324 * Ss[2,4]
        + 9.0 * t324 * Ss[4,2]
        + 78.0 * t49 * Ss[2,4]
        + 96.0 * Ss[1,0] * Ss[1,4]
        - 384.0 * Ss[1,1] * Ss[1,3]
        + 288.0 * Ss[1,2] * Ss[3,0]
    )
    t356 = Ss[1,1] * Ss[2,1]
    t359 = Ss[1,2] * Ss[2,0]
    t381 = (
        72.0 * t49 * t89 * Ss[2,0]
        - 9.0 * t115 * t277
        - 5.0 * t115 * Ss[0,6]
        + 144.0 * t174 * t356
        + 36.0 * t174 * t359
        - 648.0 * t177 * t356
        - 72.0 * t177 * t359
        - 12.0 * t191 * Ss[1,3]
        + 144.0 * t33 * t279
        - 36.0 * t49 * t279
        + t324 * Ss[0,6]
        + 13.0 * t49 * Ss[0,6]
        + 117.0 * t49 * Ss[4,2]
    )
    t410 = (
        -360.0 * N * t279
        - 48.0 * N * Ss[2,4]
        - 324.0 * t1 * t277
        - 36.0 * t1 * t279
        + 198.0 * t1 * Ss[4,2]
        + 48.0 * Ss[0,4] * t12
        + 72.0 * t180 * t356
        - 252.0 * t180 * t359
        + 225.0 * t33 * t277
        - 207.0 * t33 * Ss[4,2]
        + 2160.0 * t83 * t356
        + 720.0 * t83 * t359
        + 288.0 * Ss[1,0] * Ss[3,2]
        - 576.0 * Ss[1,1] * Ss[3,1]
    )
    t442 = (
        -360.0 * N * Ss[1,2] * Ss[3,0]
        - 36.0 * t49 * Ss[1,2] * Ss[3,0]
        - 432.0 * Ss[1,0] * Ss[1,2] * Ss[2,0]
        + 48.0 * t185 * Ss[1,3]
        + 216.0 * t185 * Ss[3,1]
        - 312.0 * t188 * Ss[1,3]
        + 672.0 * t196 * Ss[1,3]
        + 288.0 * t201 * Ss[1,2]
        - 216.0 * t238 * t89
        - 360.0 * t238 * Ss[2,2]
        - 90.0 * t246 * Ss[2,2]
        - 306.0 * t256 * Ss[2,2]
        - 1728.0 * t259 * t89
    )
    t458 = 5.0 * N
    t500 = (
        (t92 - t34 - t458 + 20.0) * Ss[0,4]
        - 12.0 * t70
        - 12.0 * (t1 - t458 + 8.0) * t89
        + 12.0 * (1.0 + N) * t12 * Ss[2,0]
        - 36.0 * t96
        + 15.0 * N * t96
        - 3.0 * t1 * t96
        + 48.0 * Ss[2,2]
        - 12.0 * N * Ss[2,2]
        - 18.0 * t1 * Ss[2,2]
        + 6.0 * t33 * Ss[2,2]
        + 3.0
        * Ss[1,0]
        * (-4.0 * (t1 - N - 4.0) * Ss[1,2] - 8.0 * t80 * Ss[3,0])
        + 12.0 * Ss[4,0]
        - 3.0 * N * Ss[4,0]
        + 3.0 * t1 * Ss[4,0]
    )
    t507 = t1 * Ss[0,4]
    t512 = t33 * Ss[0,4]
    t515 = t49 * Ss[0,4]
    t518 = t115 * Ss[1,1]
    t521 = (
        756.0 * t266 * Ss[2,2]
        + 144.0 * t33 * Ss[1,2] * Ss[3,0]
        - 36.0 * t1 * Ss[1,2] * Ss[3,0]
        + 90.0 * t235 * Ss[2,2]
        - 324.0 * t191 * Ss[3,1]
        + 18.0 * t100 * t46 * (-2.0 * t15 * t12 + (t1 - t458 + 4.0) * Ss[2,0])
        - 3.0 * t100 * Ss[0,2] * t500
        + 276.0 * t180 * Ss[1,4]
        - 360.0 * t217 * Ss[1,2]
        + 156.0 * t507 * t12
        + 1224.0 * t249 * t89
        - 72.0 * t512 * t12
        + 12.0 * t515 * t12
        - 36.0 * t518 * Ss[3,1]
    )
    t526 = t115 * Ss[1,0]
    t555 = (
        -6.0 * t115 * Ss[0,4] * Ss[2,0]
        - 18.0 * t115 * Ss[2,0] * Ss[2,2]
        + 72.0 * t49 * t12 * Ss[2,2]
        - 72.0 * t33 * t122 * Ss[1,2]
        - 576.0 * t33 * t89 * Ss[2,0]
        + 60.0 * t174 * Ss[1,4]
        + 180.0 * t174 * Ss[3,2]
        - 156.0 * t177 * Ss[1,4]
        - 78.0 * t512 * Ss[2,0]
        + 42.0 * t515 * Ss[2,0]
        - 12.0 * t518 * Ss[1,3]
        - 12.0 * t526 * Ss[1,4]
        - 36.0 * t526 * Ss[3,2]
    )
    t569 = N * Ss[0,4]
    t587 = (
        -2016.0 * N * t89 * Ss[2,0]
        + 1656.0 * t1 * t89 * Ss[2,0]
        - 144.0 * t569 * t12
        - 468.0 * t177 * Ss[3,2]
        + 828.0 * t180 * Ss[3,2]
        - 288.0 * t188 * Ss[3,1]
        + 1008.0 * t196 * Ss[3,1]
        + 792.0 * t249 * Ss[2,2]
        - 936.0 * t259 * Ss[2,2]
        - 864.0 * t269 * Ss[2,1]
        + 6.0 * t507 * Ss[2,0]
        + 84.0 * t569 * Ss[2,0]
        - 264.0 * t83 * Ss[1,4]
        - 792.0 * t83 * Ss[3,2]
    )
    t600 = t79 ** 2
    uglyD1O3E = (
        (
            -12.0 * t4 * t5 * t6
            + 36.0
            * t6
            * (
                2.0 * t16 * N * Ss[2,0]
                + N * t4 * Ss[0,2]
                - 2.0 * t4 * t12
            )
            - 24.0
            * N
            * Ss[0,1]
            * t5
            * (
                N * t27 * Ss[0,3]
                - 6.0 * t27 * Ss[1,0] * Ss[1,1]
                + 3.0 * t36 * Ss[2,1]
            )
            + 3.0
            * t5
            * (
                -6.0 * t1 * (4.0 * t1 - 21.0 * N + 29.0) * t46
                + N
                * (7.0 * t49 - 36.0 * t33 + 79.0 * t1 - t53 + 40.0)
                * Ss[0,4]
                - 12.0
                * N
                * Ss[0,2]
                * (
                    (-t58 + 42.0 * N - 58.0) * t12
                    + (t62 - 19.0 * t1 + t64 - 2.0) * Ss[2,0]
                )
                - 36.0 * t4 * t70
                + 24.0
                * N
                * (t34 - 24.0 * N + 31.0)
                * t12
                * Ss[2,0]
                - 24.0 * t82 * t83 * (2.0 * Ss[1,2] + Ss[3,0])
                + 3.0
                * N
                * (
                    -8.0 * t82 * t89
                    - 2.0 * (t92 - 9.0 * t1 + t94 + 12.0) * t96
                    + t100
                    * (
                        2.0 * (t2 - t94 + 8.0) * Ss[2,2]
                        + (t1 - N + 4.0) * Ss[4,0]
                    )
                )
            )
            - 6.0 * N * Ss[0,1] * (t144 + t194 + t231 + t272)
            + N * (t323 + t351 + t381 + t410 + t442 + t521 + t555 + t587)
        )
        / (-5.0 + N)
        / (-4.0 + N)
        / t15
        / t600
        / t81
        / N
        / 16.0
    )
    return uglyD1O3E
