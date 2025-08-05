from dataclasses import dataclass, field
import enum

import jax
import jax.numpy as jnp
from jax import grad, jit
from jax.scipy.stats import norm


class OptionType(enum.Enum):
    Call = 1
    Put = -1

def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (jnp.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    return (
        S * norm.cdf(d1) - K * jnp.exp(-r * T) * norm.cdf(d2)
        if option_type == "call"
    )


def implied_vol_newton(
    S: float,
    K: float,
    T: float,
    r: float,
    mkt_price: float,
    option_type="call",
    init_sigma=0.2,
    tol=1e-6,
    max_iter=100,
):
    """
    Calculate the implied volatility using the Newton-Raphson method.
    """
    sigma = init_sigma

    for _ in range(max_iter):
        price = black_scholes(S, K, T, r, sigma, option_type)
        vega = grad(black_scholes, argnums=4)(S, K, T, r, sigma, option_type)

        if vega == 0:
            raise ValueError("Vega is zero; cannot compute implied volatility.")

        diff = mkt_price - price
        if jnp.abs(diff) < tol:
            return sigma

        sigma += diff / vega

    raise ValueError("Implied volatility calculation did not converge.")


@dataclass
class Contract:
    """temp redo this later"""

    symbol: str
    option_type = OptionType(names="Call", )
    strike: float  # S_t
    expiration: str
    mkt_price: float
    underlying_price: float
    time_to_expiry: float  # expiry in days (T /= 365)
    is_long: bool = True
    implied_vol: float = field(default=20.0)

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
