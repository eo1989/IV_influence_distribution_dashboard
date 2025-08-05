from financetoolkit import Toolkit, options
import pandas as pd
from py_vollib_vectorized import implied_volatility
from dotenv import load_dotenv

load_dotenv()


TICKERS = ["ARM", "TSLA", "AAPL", "VKTX", "BABA", "NVDA", "PLTR", "RIVN", "LCID"]

def main():

    toolkit = Toolkit(
        tickers = TICKERS,
        api_key={
            # TODO: enter API KEYS FROM DOTENV HERE
            'alpha_vantage': ,
            'fmp': ,
        }
    )

    options_data = pd.DataFrame([
        {}
    ])
