import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
VIX=15.0; SPOT=23500; QTY=650; RUP=SPOT*QTY/100.0
strad=0.8*(VIX/100)*np.sqrt(5/252)*100
def fly(r): return 0.55*strad-np.minimum(np.abs(r),2.0)
def jb(r):
    cr=0.42*strad+0.30-0.18*strad
    return cr+np.minimum(0.0,r+2.0)-np.minimum(np.maximum(r-1.0,0.0),1.5)+np.maximum(-4.0-r,0.0)
def jbe(r):
    cr=0.42*strad+0.30-0.18*strad
    return cr-np.maximum(r-2.0,0.0)-np.minimum(np.maximum(-1.0-r,0.0),1.5)+np.maximum(r-4.0,0.0)
plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":0.25})
fig,ax=plt.subplots(1,3,figsize=(14.5,4.3),dpi=130)
fig.suptitle("NIFTY Premium-Selling — Entry Regimes (research/64): payoffs · calm-by-hold · intra-hold brakes",fontsize=12,fontweight="bold")
r=np.linspace(-6,6,600); a=ax[0]
a.axhline(0,color="#888",lw=0.8); a.axvspan(-2,2,color="#2e7d32",alpha=0.06)
a.plot(r,fly(r)*RUP,color="#1565c0",lw=2,label="Iron fly (±2% wings)")
a.plot(r,jb(r)*RUP,color="#2e7d32",lw=2,label="Bull jade + 4% put")
a.plot(r,jbe(r)*RUP,color="#c62828",lw=1.8,ls="--",label="Bear reverse-jade")
for x in (-2,2): a.axvline(x,color="#bbb",lw=0.8,ls=":")
a.set_title("Structure payoffs at expiry (per 10-lot, VIX~15 proxy)",fontsize=10.5)
a.set_xlabel("NIFTY move at expiry (%)"); a.set_ylabel("P&L (Rs)")
a.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f"{v/1000:,.0f}k")); a.legend(fontsize=8.5,loc="lower center")
b=ax[1]; x=np.arange(4); base=[77.5,68.4,59.6,39.5]; gated=[86.2,77.8,68.8,47.8]
b.bar(x-0.2,base,0.4,label="base",color="#90a4ae"); b.bar(x+0.2,gated,0.4,label="compression + VIX 13-22",color="#2e7d32")
for i,(bv,gv) in enumerate(zip(base,gated)):
    b.text(i-0.2,bv+1,f"{bv:.0f}",ha="center",fontsize=8); b.text(i+0.2,gv+1,f"{gv:.0f}",ha="center",fontsize=8,fontweight="bold")
b.set_xticks(x); b.set_xticklabels(["3d","4d","5d","8d"]); b.set_ylim(0,100)
b.set_title("Calm-survival by hold length (%)",fontsize=10.5); b.set_xlabel("hold (trading days)"); b.set_ylabel("P(no 2% breach)"); b.legend(fontsize=8.5,loc="upper right")
c=ax[2]; d3=[0.15,0.45,0.78,1.16,1.68]; p3=[90,88,85,73,48]; d4=[0.16,0.48,0.83,1.22,1.72]; p4=[99,98,92,83,64]
c.plot(d3,p3,"o-",color="#1565c0",lw=2,label="at day-3 close"); c.plot(d4,p4,"s--",color="#6a1b9a",lw=1.6,label="at day-4 close")
c.axvline(1.4,color="#e65100",lw=1.6,ls="--"); c.text(1.42,33,"~1.4% caution\nroll/close",color="#e65100",fontsize=8.5)
c.set_ylim(30,102); c.set_xlim(0,2); c.set_title("Intra-hold: P(finish calm) by drift",fontsize=10.5)
c.set_xlabel("drift from entry at day-3/4 close (%)"); c.set_ylabel("P(finish calm to day-5) %"); c.legend(fontsize=8.5,loc="lower left")
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig("/home/arun/quantifyd/frontend/public/nifty_fly_payoffs.png",bbox_inches="tight",facecolor="white")
print("saved")
