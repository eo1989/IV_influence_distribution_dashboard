import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from numpyro.contrib.control_flow import scan

# Error Models


def homoscedastic() -> None:
    """
    hyperparameters: c_0, g_0, G_0

    returns:
    sigma2 (numpy.ndarray): The variance of the errors.
    """
    # fixed hyperparameters
    c_0 = 2.5
    g_0 = 5
    G_0 = 3.33
    C_0 = numpyro.sample("C_0", dist.Gamma(g_0, G_0))
    sigma2 = numpyro.sample("sigma2", dist.InverseGamma(c_0, C_0))
    return sigma2


def stochastic_volatility(n_steps) -> None:
    """
    We estimate the log-volatility (h_t) from which we derive the heteroscedastic (sigma2_t)

    Parameters:
        n_steps (int): The number of time steps for which to propagate the error.
        hyperparameters: b_mu, a_phi, b_phi, B_mu, B_sigma

    Returns:
        sigma2 (jnp.array): The modelled variance for each time step.
    """
    # fixed hyperparameters

    b_mu = 0.0
    a_phi = 5.0
    b_phi = 1.5
    B_mu = 1.0
    B_sigma = 1.0

    sv_phi = numpyro.sample("sv_phi", dist.Beta(a_phi, b_phi))
    sv_phi_trans = (sv_phi * 2) - 1
    sv_sigma2_eta = numpyro.sample(
        "sv_sigma2_eta", dist.Gamma(0.5, 0.5 * B_sigma)
    )
    sv_mu = numpyro.sample("sv_mu", dist.Normal(b_mu, jnp.sqrt(B_mu)))
    h_0 = numpyro.sample(
        "h_0",
        dist.Normal(sv_mu, jnp.sqrt(sv_sigma2_eta / (1 - sv_phi_trans**2))),
    )
    scan_vars = (h_0,)

    def sv_ht_scan(t_carry, _) -> None:
        # sample h_t given h_t-1, params
        (h_curr,) = t_carry
        h_curr = numpyro.sample(
            "h_t",
            dist.Normal(
                sv_mu + sv_phi_trans * (h_curr - sv_mu), jnp.sqrt(sv_sigma2_eta)
            ),
        )
        t_carry = (h_curr,)
        return t_carry, h_curr

    _, h_t = scan(sv_ht_scan, scan_vars, jnp.zeros(n_steps))

    sigma2 = numpyro.deterministic("sigma2", jnp.exp(h_t))
    return sigma2
