# ignore
# %%
import math
import time
from random import random
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# %%
start_time = time.time()


def binomial_return(qty_up, up_return, up_prob, start_price, _steps):
    qty_down = _steps - qty_up
    down_prob = 1 - up_prob
    down_return = (down_prob / up_prob) * up_return
    paths = math.factorial(_steps) / (
        math.factorial(qty_down) * math.factorial(qty_up)
    )
    _prob = (up_prob**qty_up) * (down_prob**qty_down) * paths
    price = (
        start_price
        * ((1 + up_return) ** qty_up)
        * ((1 - down_return) ** qty_down)
    )
    return price, _prob


# %%
def _theo_binom_distribution(_steps, up_return, up_prob, start_price):
    _distribution = []
    for i in range(_steps + 1):
        _distribution.append(
            binomial_return(i, up_return, up_prob, start_price, _steps)
        )
    return _distribution


# %%


def _create_tree(
    start_price, _steps, up_prob, up_return, down_prob, down_return
):
    _tree = {}
    for i in range(_steps + 1):
        _tree[i] = binomial_return(i, up_return, up_prob, start_price, _steps)
    return _tree


def option_value(_tree, _strike, callput):
    _ITM = []
    for i in range(len(_tree)):
        if callput == "C" and _tree[i][0] > _strike:
            _ITM.append(_tree[i][1] * (_tree[i][0] - _strike))
        elif callput == "P" and _tree[i][0] < _strike:
            _ITM.append(_tree[i][1] * (_strike - _tree[i][0]))
    return round(sum(_ITM), 3)


def opt_delta(
    current_spot_price,
    up_prob,
    up_return,
    down_prob,
    down_return,
    _steps_dte,
    _strike,
    callput,
    current_option_value,
):
    option_up_tree = _create_tree(
        current_spot_price * (1 + up_return),
        _steps_dte,
        up_prob,
        up_return,
        down_prob,
        down_return,
    )
    option_up_val = option_value(option_up_tree, _strike, callput)

    option_down_tree = _create_tree(
        current_spot_price * (1 - down_return),
        _steps_dte,
        up_prob,
        up_return,
        down_prob,
        down_return,
    )
    option_down_val = option_value(option_down_tree, _strike, callput)

    option_up_delta = (option_up_val - current_option_value) / (
        current_spot_price * (1 + up_return) - current_spot_price
    )
    option_down_delta = (option_down_val - current_option_value) / (
        current_spot_price * (1 - down_return) - current_spot_price
    )

    option_delta = option_up_delta * up_prob + option_down_delta * down_prob
    return option_delta


def option_and_delta(
    spot_price,
    _strike,
    callput,
    _steps,
    up_prob,
    up_return,
    down_prob,
    down_return,
):
    _tree = _create_tree(
        spot_price, _steps, up_prob, up_return, down_prob, down_return
    )
    _option = option_value(_tree, _strike, callput)
    _delta = opt_delta(
        spot_price,
        up_prob,
        up_return,
        down_prob,
        down_return,
        _steps,
        _strike,
        callput,
        _option,
    )
    return _option, _delta


def randomize_stock_price_change(share_price, up_prob, up_return, down_return):
    if random() < up_prob:
        # share_price = share_price * (1 + up_return)
        share_price *= 1 + up_return
    else:
        # share_price = share_price * (1 - down_return)
        share_price *= 1 - down_return
    return round(share_price, 3)


# %%
# inputs
num_simulations = 500
sets_sims = 2
up_prob = 0.5
down_prob = 0.5
up_return = 0.1
down_return = 0.1
callput = "C"
option_position = 1
init_stock_price = 100
total_steps_dte = 10


# Initialized tables before the loop <- TODO: look for functional way to remove this loop
simulation_table = pd.DataFrame(
    columns=[
        "sim_number",
        "trial",
        "option_position",
        "_strike",
        "terminal_price",
        "stock_return",
        "delta_hedged_PnL",
    ]
)
path_table = pd.DataFrame(
    columns=[
        "sim_number",
        "trial",
        "current_step",
        "_strike",
        "option_position",
        "callput",
        "share_price",
        "cumulative_portfolio_PnL",
    ]
)
sets_of_sims_table = pd.DataFrame(
    columns=["sim_number", "sim_tool_PnL", "mean_PnL", "std_dev"]
)

for j in range(sets_sims):
    for i in range(num_simulations):
        stock_price = init_stock_price
        _strike = 110
        current_step = 0
        remaining_steps_til_expiry = total_steps_dte - current_step

        _tree = _create_tree(
            stock_price,
            remaining_steps_til_expiry,
            up_prob,
            up_return,
            down_prob,
            down_return,
        )
        option_price = option_value(_tree, _strike, callput)
        position_option_delta = opt_delta(
            stock_price,
            up_prob,
            up_return,
            down_prob,
            down_return,
            remaining_steps_til_expiry,
            _strike,
            callput,
            option_price,
        )

        # init portfolio
        position_data = {
            "current_step": current_step,
            "remaining steps_to_expiry": remaining_steps_til_expiry,
            "_strike": _strike,
            "type": callput,
            "option_position": option_position,
            "option_price": option_price,
            "option_delta": position_option_delta,
            "share_position": option_position * position_option_delta * -100,
            "share_price": stock_price,
        }

        path = {
            "sim_number": j + 1,
            "trial": 1 + j,
            "current_step": current_step,
            "_strike": _strike,
            "option_position": option_position,
            "callput": callput,
            "share_price": stock_price,
            "cumulative_portfolio_PnL": 0,
        }

        portfolio = pd.DataFrame(position_data, index=[0])

        df_path = pd.DataFrame([path])
        path_table = pd.concat([path_table, df_path], ignore_index=True)

        while remaining_steps_til_expiry > 0:
            """ ---- Simulation  ----"""

            # increment time & position data

            stock_price = randomize_stock_price_change(
                stock_price, up_prob, up_return, down_return
            )
            # current_step = current_step + 1
            current_step += 1
            remaining_steps_til_expiry -= 1
            _tree = _create_tree(
                stock_price,
                remaining_steps_til_expiry,
                up_prob,
                up_return,
                down_prob,
                down_return,
            )
            option_price = option_value(_tree, _strike, callput)
            position_option_delta = opt_delta(
                stock_price,
                up_prob,
                up_return,
                down_prob,
                down_return,
                remaining_steps_til_expiry,
                _strike,
                callput,
                option_price,
            )

            position_data = {
                "current_step": current_step,
                "remaining steps_to_expiry": remaining_steps_til_expiry,
                "_strike": _strike,
                "type": callput,
                "option_position": option_position,
                "option_price": option_price,
                "option_delta": position_option_delta,
                "share_position": option_position
                    * position_option_delta
                    * -100,
                "share_price": stock_price,
            }

            df_position = pd.DataFrame([position_data])
            portfolio = pd.concat([portfolio, df_position], ignore_index=True)

            # compute the change in share price from one step to the next
            portfolio['share_price_change'] = portfolio['share_price'].diff()
            portfolio['option_price_change'] = portfolio['option_price'].diff()

            #Compute the PnL for ea. step
            portfolio['share_PnL'] = portfolio['share_price_change'] * portfolio['share_position'].shift(1)
            portfolio['option_PnL']= 100 * portfolio['option_price_change'] * portfolio['option_position'].shift(1)

            # compute the cumulative PnL over all steps
            portfolio['cumulative_share_PnL']    = portfolio['share_PnL'].cumsum()
            portfolio['cumulative_option_PnL']   = portfolio['option_PnL'].cumsum()
            portfolio['cumulative_portfolio_PnL']= portfolio['cumulative_share_PnL'] + portfolio['cumulative_option_PnL']
            cumulative_portfolio_profit = portfolio['cumulative_portfolio_PnL'].iloc[-1]
            delta_hedged_profit = portfolio['cumulative_portfolio_PnL'].iloc[-1]

            # record trial path
            if current_step <= total_steps_dte:
                path = {
                    'sim_number': j + 1,
                    'trial': i + 1,
                    'current_step': current_step,
                    '_strike': _strike,
                    'option_position': option_position,
                    'callput': callput,
                    'share_price': stock_price,
                    'cumulative_portfolio_PnL': round(cumulative_portfolio_profit, 0)
                }

            df_path = pd.DataFrame([path])
            path_table = pd.concat([path_table, df_path], ignore_index = True)


        # Simulation Data Table
        simulation = {
            'sim_number': j + 1,
            'trial': i + 1,
            'option_position': option_position,
            '_strike': _strike,
            'terminal_price': round(stock_price, 2),
            'stock_return': round((stock_price / portfolio['share_price'].iloc[0] - 1), 3),
            'delta_hedged_PnL': round(delta_hedged_profit, 0),
        }


        df_simulation = pd.DataFrame([simulation])
        simulation_table = pd.concat([simulation_table, df_simulation], ignore_index = True)
# %%

    """There are only _steps + 1 possible terminal prices, lets see their PnL distribution for each of them."""
    grouped = simulation_table.groupby('terminal_price')['delta_hedged_PnL']

    # Calculating mu & sigma for ea. group
    sim_summary = path_table[(path_table["sim_number"] == j + 1) & (path_table["current_step"] == total_steps_dte)]
    sim_mean_profit = sim_summary["cumulative_portfolio_PnL"].mean()
    sim_total_profit = sim_summary["cumulative_portfolio_PnL"].sum()
    sim_std_dev = sim_summary["cumulative_portfolio_PnL"].std()
    log_entry = {
        'sim_number': j + 1,   # trial number
        'mean_PnL': sim_mean_profit,
        'sim_total_PnL': sim_total_profit,
        'st_dev': sim_std_dev
    }
    df_log_entry = pd.DataFrame([log_entry])
    sets_of_sims_table = pd.concat([sets_of_sims_table, df_log_entry], ignore_index=True)

    print(f"Simulation_number: {j + 1}")

    print(f"Total_profit_for_all_simulations: {round(sim_total_profit, 1)}")
    print(f"Mean_profit_for_all_simualtions: {round(sim_mean_profit, 1)}")
    print(f"St_dev_for_call_simulations: {round(sim_std_dev, 1)}")
    print(f"St_dev_scaled_to_initial_option_premium: {round(.01 * round(sim_std_dev, 0)/portfolio['option_price'].iloc[0], 3)}")
    print("---"*40)
# %%
# First figure
fig1, ax1 = plt.subplots(2, 1, figsize = (10, 10))

# Plot 1: histogram of terminal prices
sns.histplot(simulation_table['terminal_price'], kde=True, ax = ax1[0], stat="percent")
ax1[0].set_title('Distribution of Terminal Prices')
ax1[0].set_xlabel('Terminal Price')
ax1[0].set_ylabel('Percentage')
# ax1[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: "{:.0f}%".format(y)))
ax1[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))

# Plot 2: Scatter plot between Terminal Prices and Delta hedged PnL
sns.barplot(x = 'terminal_price', y = 'delta_hedged_PnL', data = simulation_table, ax = ax1[1])
ax1[1].set_title("Scatter plot between Terminal Price and Delta Hedged PnL")
ax1[1].set_xlabel('TerminalPrice')
ax1[1].set_ylabel('Delta Hedged PnL')

plt.tight_layout()
plt.show()
# %%
fig, ax = plt.subplots(figsize = (10, 7))

# group by sim_nuber & trial
grouped = path_table.groupby(['sim_number', 'trial'])

# plotting ea share price path
for (sim, trial), group in grouped:
    ax.plot(group['current_step'], group['share_price'], label = f"Sim {sim} Trial {trial}")

# lets customize the chart a bit
ax.set_title('Share Price Path for Each Simulation & Trial')
ax.set_xlabel('Step')
ax.set_ylabel('Share Price')

plt.tight_layout()
plt.show()
# %%
print(f"---"*40)
print(f"\n\n\nSummary of all simulations", "---"*40)
print(f"Sims per set: {num_simulations}")
print(f"Total sets: {sets_sims}")
print(f"---"*40)
print(f"Mean profit across all sets of sims: {round(sets_of_sims_table["mean_PnL"].mean(), 1)}")
print(f"Standard Dev of profit across all sets of sims: {round(sets_of_sims_table["sim_total_PnL"].std(), 1)}")
print(f"---"*40)
print(f"Mean_stock_return: {round(simulation_table['stock_return'].mean(), 3)}")
print(f"Modal_stock_return: {round(simulation_table['stock_return'].mode(), 3)}")

end_time = time.time()
elapsed_time = round(end_time - start_time, 0)
# %%
"""Actual, Theoretical Distributions"""
# Theoretical
distribution = _theo_binom_distribution(total_steps_dte, up_return, up_prob, init_stock_price)
distro_df = pd.DataFrame(distribution, columns=['terminal_price', 'theo_frequency'])
distro_df['theo_frequency'] = distro_df['theo_frequency'].round(4)

# Actual
terminal_price_proportions = simulation_table['terminal_price'].value_counts(normalize=True)
terminal_price_proportions = terminal_price_proportions.reset_index()
terminal_price_proportions.columns = ['terminal_price', 'actual_frequency']
terminal_price_proportions.sort_values(by='terminal_price', inplace=True)
terminal_price_proportions['actual_frequency'] = terminal_price_proportions['actual_frequency'].round(4)

# now merge the table
_decimals = 2
distro_df['terminal_price'] = distro_df['terminal_price'].round(_decimals)
terminal_price_proportions['terminal_price'] = terminal_price_proportions['terminal_price'].round(_decimals)

merged_df = distro_df.merge(terminal_price_proportions, on='terminal_price', how='left')
merged_df['actual-theo'] = merged_df['actual_frequency'] - merged_df['theo_frequency']

print(f"---"*40)
print(merged_df)
print(f"---"*40)

print(f"The code took {elapsed_time} seconds to run.")
# %%
# Second figure
fig2, ax2 = plt.subplots(2, 1, figsize = (10, 10))

# plot 3: Bar chart of actual & theoretical frequencies
# TODO There has to be a better way to do this!
position = list(range(len(merged_df['terminal_price'])))

width = 0.4
ax2[0].bar(position, merged_df['actual_frequency'], width = width, label = 'Actual Frequency', color = 'blue', edgecolor = 'gray')
ax2[0].bar([p + width for p in position], merged_df['theo_frequency'], width = width, label = 'Theoretical Frequency', color = 'black', edgecolor = 'gray')
ax2[0].set_xticks([p + 0.5 * width for p in position])
ax2[0].set_xticklabels(merged_df['terminal_price'].values, rotation = 45, ha = 'right')
ax2[0].set_title('Comparison of Actual & Theoretical Frequencies for each Terminal Price')
ax2[0].set_xlabel('Terminal Price')
ax2[0].set_ylabel('Frequency')
ax2[0].legend(['Actual Frequency', 'Theoretical Frequency'], loc = 'upper left')
ax2[0].grid(axis = 'y')
ax2[0].yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals = 0)) # format as percent with 1 decminal

# plot 4: Difference between Actual & Theoretical Frequencies
sns.barplot(x = 'terminal_price', y = 'actual-theo', data = merged_df, ax = ax2[1], color = 'blue', edgecolor = 'black')
ax2[1].set_title('Difference between Actual & Theoretical Frequencies')
ax2[1].set_xlabel('Terminal Price')
ax2[1].set_ylabel('Actual - Theoretical (%)')
ax2[1].grid(axis = 'y')
ax2[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=2)) # format as percent with 1 decimal

plt.tight_layout()
plt.show()
# %%

