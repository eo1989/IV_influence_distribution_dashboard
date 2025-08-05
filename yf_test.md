---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: IV_influence_distribution_dashboard
    language: python
    name: python3
---

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
```

```python
data = yf.download(
    "AAPL", "2020-01-01", "2025-03-18", group_by="ticker", rounding=True
).set_index(0)[]
data.head()
```

```python
data["Close"].plot()

data["Volume"].plot()

data["returns"] = data["Close"].pct_change()
data["returns"].plot()

data["cum_returns"] = (1 + data["returns"]).cumprod()
data["cum_returns"].plot()

data["moving_avg"] = data["Close"].rolling(window=5).mean()
data["moving_avg"].plot()
plt.show()
```

```python

```
