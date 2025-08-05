# ruff ignore: B007
# %%
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=4)
rng = np.random.default_rng(42)
import pandas as pd

# import polars as ps
# import jax as jx

# %%
"""Some parameters to consider"""
mu_return = 0.12
vol = 0.198
years = 97
init_bankroll = 100
num_sims = 10_000

# %%
"""Peanut Butter Jell-Simulation time"""
# return_stream = np.random.normal(mu_return, vol, years)
return_stream = rng.normal(mu_return, vol, years)

# %%
# applying ze compounding
sim_wealth = init_bankroll * (1 + return_stream).cumprod()

# %%
# we run this tow-desired number of simulations
terminal_wealth = []
for i in range(num_sims):
    # return_stream = np.random.normal(mu_return, vol, years)
    return_stream = rng.normal(mu_return, vol, years)
    sim_wealth = init_bankroll * (1 + return_stream).cumprod()
    terminal_wealth.append(sim_wealth[-1])
    plt.plot(sim_wealth)


# %%
# computanteshuns
# theoretical
median_return = mu_return - 0.5 * vol**2
theo_median = 100 * (1 + median_return) ** years
theo_mu = 100 * (1 + mu_return) ** years
theo_stdev_wealth = vol * years**0.5

# %%
# Sample your own supply
mu = np.mean(terminal_wealth)
median = np.median(terminal_wealth)
sample_stdev_wealth = np.std(terminal_wealth)

# %%
# convert terminal wealth from a list to a dataframe
wealth_results = pd.DataFrame(terminal_wealth, columns=["wealth"])
copy_wealth_results_xlsx = wealth_results.to_string(index=False)
wealth_xlsx = wealth_results.to_excel(
    "Terminal_Wealth.xlsx", index=False, engine="openpyxl"
)
wealth_results


# %%
# print that shit
print(
    f"Annual theoretical mean return = {str(mu_return)}, annual theoretical median return = {str(median_return):.6}"
)
print(f"Mean = {str(mu):.10}, theoretical mean = {str(theo_mu):.10}")
print(
    f"Median = {str(median):.10}, theoretical median = {str(theo_median):.10}"
)
print(
    f"Theoretical stdev wealth = {str(theo_stdev_wealth * theo_mu):.10}, Sample stdev wealth = {str(sample_stdev_wealth):.10}"
)
# %%
