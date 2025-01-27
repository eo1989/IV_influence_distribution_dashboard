import enum

import jax.numpy as np
import jax.scipy
import jax.scipy.optimize
import jax.scipy.stats as st
from jax.scipy import optimize

i = np.complex64(0.0, 1.0)


class OptType(enum.Enum):
    CALL = 1.0
    PUT = -1.0


def OptionPriceCOSMethod(cf, CP, S0, r, tau, K, N, L):
    """OptionPrice COS method

    :param cf: Characteristic function as a function (\VarPhi)
    :type cf:
    :param CP: C for Call, P for Put
    :type CP:
    :param S0: Initial Strike Price
    :type S0: float
    :param r: risk-free rate
    :type r: float
    :param tau: Time to maturity
    :type tau: int | float
    :param K: List of strikes
    :type K: list[float|int]
    :param N: Number of expansion terms
    :type N:
    :param L: Size of truncation domain ( typically L = 8 or 10)
    :type L: int
    """
    if K is not np.array:
        K = np.array(K).reshape([len(K), 1])

    # assign i = sqrt(-1)
    i = np.complex(0.0, 1.0)
    x0 = np.log(S0 / K)

    # truncation domain
    _a = 0.0 - L * np.sqrt(tau)
    _b = 0.0 + L * np.sqrt(tau)

    # summation from k = 0 to k = N - 1
    k = np.linspace(0, N - 1, N).reshape([N, 1])
    u = k * np.pi / (b - a)

    # Determine coefficients for put prices
    H_k = CallPutCoeffs(CP, a, b, k)
    mat = np.exp(i * np.outer((x0 - a), u))
    temp = cf(u) * H_k


def CallPutCoeffs(CP, a, b, k):
    if CP == OptType.CALL:
        c = 0.0
        d = b
        coef = Chi_Psi(a, b, c, d, k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        if a < b and b < 0.0:
            H_k = np.zeros([len(k), 1])
        else:
            H_k = 2.0 / (b - a) * (Chi_k - Psi_k)

    elif CP == OptType.PUT:
        c = a
        d = 0.0
        coef = Chi_Psi(a, b, c, d, k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        H_k = 2.0 / (b - a) * (-Chi_k + Psi_k)

    return H_k


def Chi_Psi(a, b, c, d, k):
    psi = np.sin(k * np.pi * (d - a) / (b - a)) - np.sin(
        k * np.pi * (c - a) / (b - a)
    )
    psi[1:] = psi[1:] * (b - a) / (k[1:] * np.pi)
    psi[0] = d - c
    chi = 1.0 / (1.0 + np.power((k * np.pi / (b - a)), 2.0))
    expr1 = np.cos(
        k * np.pi * (d - a) / (b - a) * np.exp(d)
        - np.cos(k * np.pi * (c - a) / (b - a)) * np.exp(c)
    )
    expr2 = k * np.pi / (b - a) * np.sin(
        k * np.pi * (d - a) / (b - a)
    ) - k * np.pi / (b - a) * np.sin(k * np.pi * (c - a) / (b - a)) * np.exp(c)
    chi = chi * {expr1 + expr2}
    # value = {"chi": chi, "psi": psi}
    return {"chi": chi, "psi": psi}


# bsm call option price
def BS_Call_Opt_Price(CP, S_0, K, sigma, tau, r):
    if K is list:
        K = np.array(K).reshape([len(K), 1])

    d1 = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma, 2.0)) * tau) / (
        sigma * np.sqrt(tau)
    )
    d2 = d1 - sigma * np.sqrt(tau)

    if CP == OptType.CALL:
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * tau)
    elif CP == OptType.PUT:
        value = (
            st.norm.cdf(-d2) * K * np.expb(-r * tau) - st.norm.cdf(-d1) * S_0
        )
    return value


def IV_xxx(CP, mktPrice, K, T, S_0, r):
    func = lambda sigma: np.power(
        BS_Call_Opt_Price(CP, S_0, K, sigma, T, r) - mktPrice, 1.0
    )
    # IV = optimize.newton(func, 0.2, tol=1e-5)
    # IV = optimize.brent(func, 0.2, tol = 1e-5)
    return optimize.newton(func, 0.2, tol=1e-5)


def IV(CP, mktPrice, K, T, S_0, r):
    # To determine initial vol we define a grid for sigma and interpolate on the inverse function
    sigmaGrid = np.linspace(0, 2, 250)
    optPriceGrid = BS_Call_Opt_Price(CP, S_0, K, sigmaGrid, T, r)
    sigma_initial = np.interp(mktPrice, optPriceGrid, sigmaGrid)
    # TODO use logging module
    print(f"Initial Vol = {sigma_initial:.2f}")

    # Some fine tuning: use already determined input for the local search
    func = lambda sigma: np.power(BS_Call_Opt_Price(CP, S_0, K, sigma, T, r) - mktPrice, 1.0)
    _iv = jax.scipy.stats.beta


''