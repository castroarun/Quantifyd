#!/usr/bin/env python3
import sys, json
sys.path.insert(0,"/home/arun/quantifyd")
from kiteconnect import KiteConnect
from config import KITE_API_KEY
tok=json.load(open("/home/arun/quantifyd/backtest_data/access_token.json"))["access_token"]
k=KiteConnect(api_key=KITE_API_KEY); k.set_access_token(tok)
inst=k.instruments("NFO")
recfut=[i for i in inst if i["instrument_type"]=="FUT" and "REC" in i["tradingsymbol"].upper()]
recfut.sort(key=lambda x:str(x["expiry"]))
print("matches:",len(recfut))
for i in recfut[:5]:
    print(i["tradingsymbol"], "name=",i["name"], "lot_size=",i["lot_size"], "expiry=",i["expiry"])
