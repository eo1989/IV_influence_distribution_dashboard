from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax import grad, jit
from jax.scipy.stats import norm


def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (jnp.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    return (
        S * norm.cdf(d1) - K * jnp.exp(-r * T) * norm.cdf(d2)
        if option_type == "call"
        else K * jnp.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    )


def implied_vol_newton(
    S,
    K,
    T,
    r,
    mkt_price,
    option_type="call",
    init_sigma=0.2,
    tol=1e-6,
    max_iter=100,
):
    return


@dataclass
class Contract:
    """temp redo this later"""

    symbol: str
    option_type: str  # "call" or "put"
    strike: float  # S_t
    expiration: str
    mkt_price: float
    underlying_price: float
    time_to_expiry: float  # expiry in days (T /= 365)
    is_long: bool = True
    implied_vol: float = field(default=0.0)

    def compute_iv(self, rfr: float = 0.01, init_sigma: float = 0.20):
        """
        Use JAX to calculate the contracts implied_vol using the Newton-Raphson solver
        and store it in self.implied_vol
        """
        self.implied_vol = implied_vol_newton(
            S=self.underlying_price,
            K=self.strike,
            T=self.time_to_expiry,
            r=rfr,
            mkt_price=self.mkt_price,
            option_type=self.option_type,
            init_sigma=init_sigma,
        )
