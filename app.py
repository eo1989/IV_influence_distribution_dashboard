# %%
# import datetime

# import dash
# import plotly.graph_objects as go

import yfinance as ff
from datetime import datetime
import pandas as pd
import numpy as np
# from dash import Input, Output, dcc, html

# app = dash.Dash(__name__)
# server = app.server

# %%
def fetch_options_iv(ticker, start_date = None, end_date = None):
    stock = ff.Ticker(ticker)
    expirations = stock.options
    if not expirations:
        return None

    # TODO this assumes that the first expiration is the front month
    front_month = expirations[0]
    options = stock.option_chain(front_month)
    calls = options.calls
    puts = options.puts

    avg_call_iv = calls['impliedVolatility'].mean() * 100 if not calls.empty else None
    avg_put_iv = puts['impliedVolatility'].mean() * 100 if not puts.empty else None

    return {'calls': avg_call_iv, 'puts': avg_put_iv}





# %%
