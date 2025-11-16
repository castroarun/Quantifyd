import pandas_datareader as web
import pandas as pd
import yfinance as yf
from datetime import date
from nselib import capital_market as em

qlist = em.equity_list()
print(qlist.shape)

def main():
    pass


if __name__ == "__main__":
    main()