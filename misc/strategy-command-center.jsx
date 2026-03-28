import { useState, useEffect } from "react";

// ─── DATA ────────────────────────────────────────────────────
const BLUEPRINTS = [
  { id:"s1",name:"Momentum Breakout",version:"v2.4",github:"https://github.com/arun-castronix/momentum-breakout",lastCommit:"18 Mar 2026",commitMsg:"fix: volume filter threshold",
    systemType:"Swing",holdingPeriod:"2–7 days",timeframe:"Daily",instruments:"Nifty 500 Equities",capital:"₹4.5L",
    description:"Identifies stocks breaking out of consolidation with volume confirmation. Uses 20DMA trend filter and ATR volatility gating.",
    entry:["Price closes above 20-day high, volume > 1.5× 20d avg","Stock above 20 DMA (trend filter)","ATR(14) within 2%–5% of price","RSI(14) between 55–75","Sector RS rank top 3 of 11"],
    exit:["Target: 3× ATR from entry (R:R = 3:1)","Time: Close after 7 days if target not hit","Trailing stop after 1.5× ATR profit","Exit if sector RS drops below 6"],
    stopLoss:["Initial: 1.5× ATR below entry","Hard: -4% absolute max","Trailing: Breakeven after 1× ATR profit","Gap-down: Exit at open if gap > 3%"],
    positionSizing:"Risk 1% per trade. Max 5 concurrent.",
    indicators:["EMA(20)","ATR(14)","RSI(14)","Vol SMA(20)","Sector RS"],filters:["MCap > ₹5K Cr","Avg vol > ₹10 Cr","Not in F&O ban"],tags:["breakout","momentum"],
    backtest:{period:"Jan 2022 – Mar 2026",totalTrades:487,winRate:62.4,avgWin:3.8,avgLoss:-1.9,profitFactor:1.92,expectancy:1.42,maxDD:-11.2,avgDD:-4.1,maxConsecLoss:5,avgHolding:"3.4d",sharpe:1.85,sortino:2.31,calmar:1.65,bestMonth:8.2,worstMonth:-4.8,avgMonthly:2.1,maxDDTrade:-4.0,avgDDTrade:-1.6}},
  { id:"s2",name:"Mean Reversion F&O",version:"v3.1",github:"https://github.com/arun-castronix/mean-reversion-fno",lastCommit:"19 Mar 2026",commitMsg:"feat: Bollinger squeeze pre-filter",
    systemType:"Intraday",holdingPeriod:"1–6 hours",timeframe:"15min",instruments:"NIFTY & BANKNIFTY Options",capital:"₹6L",
    description:"Trades reversals at oversold levels using Keltner Channels and RSI divergence on 15-min charts.",
    entry:["Price touches lower Keltner (2.0 ATR, 20 EMA)","RSI(14) < 30 with bullish divergence","Within ±0.5% of known support","VIX > 13","Only 09:30–14:30 IST"],
    exit:["Target: Middle Keltner band (20 EMA)","Time: Exit by 15:00 IST","Quick: 50% at 1× risk, 50% run","Exit if VIX drops > 2 pts"],
    stopLoss:["Below swing low","Max: Premium × 40%","0.5% adverse in 15 min → exit","No averaging down"],
    positionSizing:"Max 2 lots. Max 2 concurrent. Risk 1.5%.",
    indicators:["Keltner(20,2.0)","RSI(14)","VWAP","VIX","OI Change"],filters:["ATM±1 strike","Spread<₹5","OI>5L","No expiry day"],tags:["mean-reversion","options","intraday"],
    backtest:{period:"Apr 2023 – Mar 2026",totalTrades:896,winRate:58.1,avgWin:5.2,avgLoss:-3.1,profitFactor:1.68,expectancy:0.83,maxDD:-14.6,avgDD:-6.8,maxConsecLoss:7,avgHolding:"2.8h",sharpe:1.42,sortino:1.89,calmar:1.21,bestMonth:12.4,worstMonth:-7.2,avgMonthly:2.8,maxDDTrade:-3.1,avgDDTrade:-1.4}},
  { id:"s3",name:"Keltner Squeeze",version:"v1.8",github:"https://github.com/arun-castronix/keltner-squeeze",lastCommit:"15 Mar 2026",commitMsg:"refactor: Pine→Python",
    systemType:"Intraday",holdingPeriod:"30min–4h",timeframe:"15min",instruments:"NIFTY & BANKNIFTY Options",capital:"₹3L",
    description:"Captures explosive moves when BB squeeze inside Keltner Channels (TTM Squeeze).",
    entry:["BB(20,2.0) inside Keltner(20,1.5)","Squeeze fires (red→green)","Histogram direction → calls/puts","First bar after fire","Min 6 squeeze bars"],
    exit:["Histogram reverses","Target: 2× ATR","No follow-through in 2h","Squeeze re-engages"],
    stopLoss:["Below/above squeeze range","Max 30% premium loss","Breakeven after 1× ATR","Hard stop past mid-Keltner"],
    positionSizing:"Risk 1%. Max 1 trade. A+ only.",
    indicators:["BB(20,2.0)","Keltner(20,1.5)","TTM Histogram","ATR(14)"],filters:["09:45–14:00 IST","VIX<22","Min 6 bars"],tags:["squeeze","volatility-breakout"],
    backtest:{period:"Jul 2023 – Mar 2026",totalTrades:312,winRate:55.1,avgWin:7.8,avgLoss:-4.2,profitFactor:1.52,expectancy:1.41,maxDD:-16.8,avgDD:-8.1,maxConsecLoss:6,avgHolding:"1.9h",sharpe:1.12,sortino:1.54,calmar:0.89,bestMonth:14.2,worstMonth:-9.1,avgMonthly:1.9,maxDDTrade:-4.2,avgDDTrade:-2.0}},
  { id:"s4",name:"Sector Rotation",version:"v1.2",github:"https://github.com/arun-castronix/sector-rotation",lastCommit:"17 Mar 2026",commitMsg:"updated weights",
    systemType:"Positional",holdingPeriod:"1–4 weeks",timeframe:"Weekly",instruments:"Nifty 500 leaders",capital:"₹5L",
    description:"Rotates into top sectors by RS ranking. Top 2 stocks from top 3 sectors.",
    entry:["Rank 11 sectors by 3M RS","Top 3 sectors","Top 2 per sector by RS+fundamentals","Weekly close > 10 WMA","Nifty > 40 WMA"],
    exit:["Sector leaves top 5 RS","Stock < 10 WMA weekly","Rebalance Fridays","Nifty < 40 WMA → exit all"],
    stopLoss:["8% initial","10 WMA trailing","Sector RS drops 4+ ranks","5% portfolio DD max"],
    positionSizing:"Equal weight 6 positions. Max 6.",
    indicators:["RS(custom)","10 WMA","40 WMA","Mansfield RS"],filters:["MCap>₹10KCr","ROE>15%","D/E<1"],tags:["sector-rotation","positional"],
    backtest:{period:"Jan 2022 – Mar 2026",totalTrades:198,winRate:67.2,avgWin:5.1,avgLoss:-3.2,profitFactor:2.18,expectancy:2.21,maxDD:-8.4,avgDD:-3.5,maxConsecLoss:3,avgHolding:"12.6d",sharpe:2.1,sortino:2.78,calmar:2.45,bestMonth:6.8,worstMonth:-3.2,avgMonthly:2.4,maxDDTrade:-3.2,avgDDTrade:-1.3}},
  { id:"s5",name:"Gap Fill Scalper",version:"v0.9β",github:"https://github.com/arun-castronix/gap-fill-scalper",lastCommit:"20 Mar 2026",commitMsg:"hotfix: volume filter",
    systemType:"Intraday",holdingPeriod:"5–45min",timeframe:"5min",instruments:"Nifty 50",capital:"₹2L",
    description:"Counter-trend on gaps >1.5% expecting fill within first hour.",
    entry:["Gap >1.5%","Wait 1st 5min candle","Wick>60% body","Vol>2× avg","Into S/R zone"],
    exit:["Target: prev close","50% at half fill","Exit 10:30 AM","Gap extends 0.5%+"],
    stopLoss:["Beyond gap extreme","Max 1%","2nd candle extends→exit","No re-entry same day"],
    positionSizing:"Risk 0.5% (beta). Max 2.",
    indicators:["Gap%","VWAP","Vol SMA(20)","Pivot S/R"],filters:["Nifty 50 only","No result days"],tags:["gap-fill","scalping","beta"],
    backtest:{period:"Oct 2024 – Mar 2026",totalTrades:420,winRate:44.3,avgWin:2.1,avgLoss:-2.4,profitFactor:0.82,expectancy:-0.42,maxDD:-18.6,avgDD:-12.3,maxConsecLoss:9,avgHolding:"18min",sharpe:0.65,sortino:0.71,calmar:0.38,bestMonth:4.2,worstMonth:-8.9,avgMonthly:-0.6,maxDDTrade:-2.4,avgDDTrade:-1.2}},
  { id:"s6",name:"Covered Call Writer",version:"v2.0",github:"https://github.com/arun-castronix/covered-call-writer",lastCommit:"14 Mar 2026",commitMsg:"auto-roll ITM",
    systemType:"Monthly",holdingPeriod:"Expiry to expiry",timeframe:"Daily",instruments:"F&O stocks + CE",capital:"₹8L",
    description:"Sells OTM covered calls on strong F&O stocks for 1.5–2.5% monthly premium yield.",
    entry:["Buy underlying cash","F&O-eligible lot","ROE>12%, QoQ+","10DMA>50DMA","Sell CE delta 0.25–0.30"],
    exit:["Expire→roll next month","Assignment→re-enter","Roll if premium<10%","Thesis breaks→close"],
    stopLoss:["Underlying -10%","Roll down if -5%","Below 200DMA weekly→exit","No SL on call"],
    positionSizing:"1 lot/stock. Max 3. ~₹2.5L each.",
    indicators:["Delta(.25-.30)","IV Pctl(>40%)","10/50/200 DMA"],filters:["Monthly CE","IVR>30%","No pre-result sells"],tags:["income","covered-call","monthly"],
    backtest:{period:"Jan 2023 – Mar 2026",totalTrades:94,winRate:78.7,avgWin:2.8,avgLoss:-4.1,profitFactor:2.64,expectancy:1.32,maxDD:-5.2,avgDD:-2.1,maxConsecLoss:2,avgHolding:"24d",sharpe:2.8,sortino:3.42,calmar:3.85,bestMonth:4.1,worstMonth:-2.8,avgMonthly:1.9,maxDDTrade:-4.1,avgDDTrade:-1.8}},
];

const INIT_STRATS = [
  {id:"s1",name:"Momentum Breakout",type:"Equity",status:"running",capital:450000,deployed:380000,pnlToday:12450,pnlTotal:87320,winRate:62,trades:148,maxDD:-4.2,sharpe:1.85,lastSignal:"2m ago",risk:"low"},
  {id:"s2",name:"Mean Reversion F&O",type:"F&O",status:"running",capital:600000,deployed:520000,pnlToday:-3200,pnlTotal:145600,winRate:58,trades:312,maxDD:-6.8,sharpe:1.42,lastSignal:"45s",risk:"medium"},
  {id:"s3",name:"Keltner Squeeze",type:"F&O",status:"paused",capital:300000,deployed:0,pnlToday:0,pnlTotal:34500,winRate:55,trades:89,maxDD:-8.1,sharpe:1.12,lastSignal:"3h",risk:"low"},
  {id:"s4",name:"Sector Rotation",type:"Equity",status:"running",capital:500000,deployed:410000,pnlToday:8700,pnlTotal:62100,winRate:67,trades:56,maxDD:-3.5,sharpe:2.1,lastSignal:"15m",risk:"low"},
  {id:"s5",name:"Gap Fill Scalper",type:"Equity",status:"error",capital:200000,deployed:185000,pnlToday:-8900,pnlTotal:-12400,winRate:44,trades:420,maxDD:-12.3,sharpe:0.65,lastSignal:"1m",risk:"high"},
  {id:"s6",name:"Covered Call Writer",type:"F&O",status:"running",capital:800000,deployed:720000,pnlToday:2100,pnlTotal:198000,winRate:78,trades:94,maxDD:-2.1,sharpe:2.8,lastSignal:"1h",risk:"low"},
];

const INIT_POS = [
  {id:"p1",strategy:"Momentum Breakout",symbol:"RELIANCE",type:"EQ",side:"LONG",qty:50,entry:2845.60,current:2892.30,pnl:2335,pnlPct:1.64,sl:2800,tp:2950,flags:[]},
  {id:"p2",strategy:"Mean Reversion F&O",symbol:"NIFTY 23500CE",type:"OPT",side:"LONG",qty:75,entry:185.50,current:212.40,pnl:2017,pnlPct:14.5,sl:150,tp:260,flags:[]},
  {id:"p3",strategy:"Mean Reversion F&O",symbol:"BNKNIFTY 50000PE",type:"OPT",side:"SHORT",qty:30,entry:320,current:345.80,pnl:-774,pnlPct:-8.06,sl:380,tp:240,flags:["Approaching SL (₹380)"]},
  {id:"p4",strategy:"Sector Rotation",symbol:"HDFCBANK",type:"EQ",side:"LONG",qty:100,entry:1685.20,current:1712.50,pnl:2730,pnlPct:1.62,sl:1650,tp:1760,flags:[]},
  {id:"p5",strategy:"Gap Fill Scalper",symbol:"INFY",type:"EQ",side:"SHORT",qty:200,entry:1520.30,current:1548.60,pnl:-5660,pnlPct:-1.86,sl:1560,tp:1480,flags:["Loss > ₹5K","Near SL (₹1560)"]},
  {id:"p6",strategy:"Covered Call Writer",symbol:"SBIN",type:"EQ",side:"LONG",qty:500,entry:785.40,current:792.10,pnl:3350,pnlPct:0.85,sl:760,tp:null,flags:[]},
  {id:"p7",strategy:"Covered Call Writer",symbol:"SBIN 800CE",type:"OPT",side:"SHORT",qty:500,entry:12.50,current:9.80,pnl:1350,pnlPct:21.6,sl:18,tp:5,flags:[]},
];

const TRADE_HIST = [
  {id:"t1",date:"20 Mar",strategy:"Momentum Breakout",symbol:"WIPRO",side:"LONG",entry:452.30,exit:468.90,pnl:4980,pnlPct:3.67,holding:"2d 4h",notes:"Clean breakout. Volume confirmed.",journal:"Waited patiently for the setup. Textbook execution."},
  {id:"t2",date:"19 Mar",strategy:"Mean Reversion F&O",symbol:"NIFTY 23400CE",side:"LONG",entry:145,exit:198.50,pnl:4012,pnlPct:36.9,holding:"4h 20m",notes:"Keltner bounce.",journal:"Confirmation candle saved me from a false entry the bar before."},
  {id:"t3",date:"19 Mar",strategy:"Gap Fill Scalper",symbol:"BAJFINANCE",side:"SHORT",entry:7250,exit:7310,pnl:-1500,pnlPct:-0.83,holding:"22m",notes:"Gap didn't fill. Volume weak.",journal:"FOMO trade. Volume wasn't there. Must add hard volume filter."},
  {id:"t4",date:"18 Mar",strategy:"Sector Rotation",symbol:"ICICIBANK",side:"LONG",entry:1245.60,exit:1289.30,pnl:8740,pnlPct:3.51,holding:"3d 2h",notes:"Banking momentum.",journal:""},
  {id:"t5",date:"18 Mar",strategy:"Covered Call Writer",symbol:"ITC 460CE",side:"SHORT",entry:8.50,exit:0.05,pnl:4225,pnlPct:99.4,holding:"7d",notes:"Full premium captured.",journal:"The boring trades make money. Love it."},
  {id:"t6",date:"17 Mar",strategy:"Keltner Squeeze",symbol:"NIFTY PE",side:"LONG",entry:95,exit:142,pnl:3525,pnlPct:49.5,holding:"2h 15m",notes:"Textbook squeeze.",journal:"Trust the system. When it's A+ grade, size up."},
  {id:"t7",date:"17 Mar",strategy:"Momentum Breakout",symbol:"LTIM",side:"LONG",entry:5120,exit:5045,pnl:-3750,pnlPct:-1.46,holding:"1d 6h",notes:"False breakout. IT weak.",journal:"Review: should I avoid IT during global risk-off?"},
  {id:"t8",date:"14 Mar",strategy:"Sector Rotation",symbol:"ICICIBANK",side:"LONG",entry:1230,exit:1269.40,pnl:4800,pnlPct:3.2,holding:"5d",notes:"Weekly rebalance winner.",journal:""},
];

const DAY_DATA = [
  {date:"20 Mar 2026",dow:"Thu",pnl:12050,trades:9,wins:6,losses:3,best:4980,worst:-5660,capUsed:1680000,
   items:[{sym:"WIPRO",str:"Momentum Breakout",side:"LONG",pnl:4980,pp:3.67,en:452.3,ex:468.9,time:"09:22–14:45",note:"Clean breakout."},{sym:"NIFTY 23500CE",str:"Mean Reversion F&O",side:"LONG",pnl:2017,pp:14.5,en:185.5,ex:212.4,time:"10:05–13:20",note:"Keltner bounce."},{sym:"HDFCBANK",str:"Sector Rotation",side:"LONG",pnl:2730,pp:1.62,en:1685.2,ex:1712.5,time:"09:45–15:10",note:"Banking RS strong."},{sym:"INFY",str:"Gap Fill Scalper",side:"SHORT",pnl:-5660,pp:-1.86,en:1520.3,ex:1548.6,time:"09:18–09:40",note:"Gap extended. Weak volume."},{sym:"RELIANCE",str:"Momentum Breakout",side:"LONG",pnl:2335,pp:1.64,en:2845.6,ex:2892.3,time:"09:22–open",note:"Holding."},{sym:"SBIN 800CE",str:"Covered Call Writer",side:"SHORT",pnl:1350,pp:21.6,en:12.5,ex:9.8,time:"09:31–open",note:"Decaying."},{sym:"TATAMOTORS",str:"Momentum Breakout",side:"LONG",pnl:2010,pp:1.96,en:685,ex:698.4,time:"11:02–open",note:"Breakout holding."},{sym:"BNKNIFTY PE",str:"Mean Reversion F&O",side:"SHORT",pnl:-774,pp:-8.06,en:320,ex:345.8,time:"10:15–open",note:"Near SL."},{sym:"TCS",str:"Sector Rotation",side:"LONG",pnl:1074,pp:0.91,en:3920,ex:3955.8,time:"09:50–open",note:"Mild."}]},
  {date:"19 Mar 2026",dow:"Wed",pnl:22100,trades:7,wins:5,losses:2,best:8740,worst:-1500,capUsed:1520000,
   items:[{sym:"NIFTY 23400CE",str:"Mean Reversion F&O",side:"LONG",pnl:4012,pp:36.9,en:145,ex:198.5,time:"10:10–14:30",note:"Perfect reversal."},{sym:"BAJFINANCE",str:"Gap Fill Scalper",side:"SHORT",pnl:-1500,pp:-0.83,en:7250,ex:7310,time:"09:16–09:38",note:"Gap extended."},{sym:"ICICIBANK",str:"Sector Rotation",side:"LONG",pnl:8740,pp:3.51,en:1245.6,ex:1289.3,time:"09:30–15:15",note:"Banking momentum."},{sym:"SBIN",str:"Covered Call Writer",side:"LONG",pnl:2850,pp:0.72,en:782,ex:787.6,time:"all day",note:"Steady."},{sym:"BN 49800CE",str:"Mean Reversion F&O",side:"LONG",pnl:3200,pp:28.1,en:220,ex:281.8,time:"11:05–14:15",note:"Oversold bounce."},{sym:"AXISBANK",str:"Sector Rotation",side:"LONG",pnl:3100,pp:2.8,en:1105,ex:1135.9,time:"09:45–15:10",note:"Rotation pick."},{sym:"TATASTEEL",str:"Gap Fill Scalper",side:"LONG",pnl:1698,pp:1.2,en:141.5,ex:143.2,time:"09:18–09:52",note:"Clean fill."}]},
  {date:"18 Mar 2026",dow:"Tue",pnl:-4200,trades:5,wins:1,losses:4,best:4225,worst:-3750,capUsed:1350000,
   items:[{sym:"LTIM",str:"Momentum Breakout",side:"LONG",pnl:-3750,pp:-1.46,en:5120,ex:5045,time:"09:30–next day",note:"False breakout."},{sym:"ITC 460CE",str:"Covered Call Writer",side:"SHORT",pnl:4225,pp:99.4,en:8.5,ex:0.05,time:"expiry",note:"Full premium."},{sym:"NIFTY PE",str:"Keltner Squeeze",side:"LONG",pnl:-1200,pp:-15.8,en:95,ex:80,time:"10:20–12:00",note:"No follow-through."},{sym:"WIPRO",str:"Momentum Breakout",side:"LONG",pnl:-2100,pp:-1.1,en:458,ex:452.9,time:"10:15–14:30",note:"Market weak."},{sym:"SBIN",str:"Covered Call Writer",side:"LONG",pnl:-1375,pp:-0.35,en:789,ex:786.2,time:"all day",note:"Minor pullback."}]},
  {date:"17 Mar 2026",dow:"Mon",pnl:18500,trades:8,wins:6,losses:2,best:5625,worst:-1800,capUsed:1720000,
   items:[{sym:"NIFTY PE",str:"Keltner Squeeze",side:"LONG",pnl:3525,pp:49.5,en:95,ex:142,time:"10:30–12:45",note:"Textbook."},{sym:"BN 49500CE",str:"Mean Reversion F&O",side:"LONG",pnl:5625,pp:35.7,en:210,ex:285,time:"10:00–15:00",note:"Oversold bounce."},{sym:"HDFCBANK",str:"Sector Rotation",side:"LONG",pnl:4200,pp:2.5,en:1670,ex:1711.75,time:"09:30–15:15",note:"Strong sector."},{sym:"RELIANCE",str:"Momentum Breakout",side:"LONG",pnl:3100,pp:2.1,en:2820,ex:2879.2,time:"09:25–14:50",note:"Volume breakout."},{sym:"TATAMOTORS",str:"Momentum Breakout",side:"LONG",pnl:2450,pp:1.8,en:672,ex:684.1,time:"11:00–15:10",note:"Auto pick."},{sym:"INFY",str:"Gap Fill Scalper",side:"SHORT",pnl:-1800,pp:-0.9,en:1535,ex:1548.8,time:"09:16–09:35",note:"False."},{sym:"SBIN",str:"Covered Call Writer",side:"LONG",pnl:1900,pp:0.48,en:780,ex:783.7,time:"all day",note:"Steady."},{sym:"TCS",str:"Sector Rotation",side:"LONG",pnl:-500,pp:-0.3,en:3940,ex:3928.2,time:"09:45–15:10",note:"Drag."}]},
  {date:"14 Mar 2026",dow:"Fri",pnl:12800,trades:6,wins:5,losses:1,best:4800,worst:-900,capUsed:1450000,
   items:[{sym:"ICICIBANK",str:"Sector Rotation",side:"LONG",pnl:4800,pp:3.2,en:1230,ex:1269.4,time:"09:30–15:15",note:"Rebalance winner."},{sym:"SBIN 790CE",str:"Covered Call Writer",side:"SHORT",pnl:3200,pp:85,en:7.5,ex:1.1,time:"week",note:"Near full decay."},{sym:"RELIANCE",str:"Momentum Breakout",side:"LONG",pnl:2800,pp:1.9,en:2810,ex:2863.4,time:"09:20–14:00",note:"Continued."},{sym:"HDFCBANK",str:"Sector Rotation",side:"LONG",pnl:1500,pp:0.9,en:1665,ex:1680,time:"09:30–15:15",note:"Steady."},{sym:"WIPRO",str:"Momentum Breakout",side:"LONG",pnl:1400,pp:1.1,en:445,ex:449.9,time:"10:05–15:00",note:"Mild."},{sym:"BAJFINANCE",str:"Gap Fill Scalper",side:"LONG",pnl:-900,pp:-0.5,en:7180,ex:7144,time:"09:17–09:45",note:"Weak signal."}]},
];

// ─── UTILS ───────────────────────────────────────────────────
const fmt = n => { if (n == null) return "—"; const a = Math.abs(n); if (a >= 100000) return (n<0?"-":"") + "₹" + (a/100000).toFixed(2) + "L"; return (n<0?"-":"") + "₹" + a.toLocaleString("en-IN"); };
const pc = n => (n > 0 ? "+" : "") + n.toFixed(2) + "%";
const cl = n => n > 0 ? "text-emerald-700" : n < 0 ? "text-red-600" : "text-slate-500";
const SB = ({s}) => { const m = {running:["bg-emerald-50","text-emerald-700","bg-emerald-500","Live"],paused:["bg-amber-50","text-amber-700","bg-amber-400","Paused"],error:["bg-red-50","text-red-700","bg-red-500","Error"],stopped:["bg-slate-100","text-slate-500","bg-slate-400","Off"]}; const [bg,tx,dt,lb] = m[s]||m.stopped; return <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-semibold ${bg} ${tx}`}><span className={`w-1.5 h-1.5 rounded-full ${dt} ${s==="running"?"animate-pulse":""}`}/>{lb}</span>; };
const RB = ({r}) => <span className={`px-1.5 py-0.5 rounded text-xs font-bold uppercase ${r==="low"?"bg-emerald-100 text-emerald-800":r==="medium"?"bg-amber-100 text-amber-800":"bg-red-100 text-red-800"}`}>{r}</span>;
const SyB = ({t}) => <span className={`px-2 py-0.5 rounded-full text-xs font-bold ${t==="Intraday"?"bg-violet-100 text-violet-800":t==="Swing"?"bg-sky-100 text-sky-800":t==="Positional"?"bg-teal-100 text-teal-800":"bg-rose-100 text-rose-800"}`}>{t}</span>;
const MC = ({label,value,sub,accent}) => <div className="bg-white rounded-xl border border-slate-200 p-3.5 flex flex-col gap-0.5 shadow-sm"><span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">{label}</span><span className={`text-xl font-bold tracking-tight ${accent||"text-slate-900"}`}>{value}</span>{sub && <span className="text-xs text-slate-500">{sub}</span>}</div>;

const MiniBar = ({data, height=56}) => {
  const max = Math.max(...data.map(d => Math.abs(d.pnl)||1)); const w = 100/data.length;
  return <svg viewBox={`0 0 100 ${height}`} className="w-full" style={{height}}>
    {data.map((d,i) => { const h2 = (Math.abs(d.pnl)/max)*(height/2-4); const y = d.pnl>=0 ? height/2-h2 : height/2; return <g key={i}><rect x={i*w+w*.15} y={y} width={w*.7} height={Math.max(h2,1)} rx="2" fill={d.pnl>=0?"#059669":"#dc2626"} opacity=".85"/><text x={i*w+w/2} y={height-1} textAnchor="middle" fontSize="5" fill="#64748b" fontFamily="monospace">{d.day||d.month}</text></g>; })}
    <line x1="0" y1={height/2} x2="100" y2={height/2} stroke="#cbd5e1" strokeWidth=".4" strokeDasharray="2,2"/>
  </svg>;
};

const PNL_W = [{day:"Mon",pnl:18500},{day:"Tue",pnl:-4200},{day:"Wed",pnl:22100},{day:"Thu",pnl:12050},{day:"Fri",pnl:0}];
const PNL_M = [{month:"Oct",pnl:45000},{month:"Nov",pnl:62000},{month:"Dec",pnl:-12000},{month:"Jan",pnl:78000},{month:"Feb",pnl:91000},{month:"Mar",pnl:61250}];

const TABS = [
  {id:"dash",label:"Dashboard",em:"📈"},
  {id:"positions",label:"Positions & Trades",em:"📊"},
  {id:"blueprints",label:"Blueprints",em:"📐"},
  {id:"deepdive",label:"Day Deep Dive",em:"🔍"},
];

// ═══════════════════════════════════════════════════════════════
export default function App() {
  const [tab, setTab] = useState("dash");
  const [strats, setStrats] = useState(INIT_STRATS);
  const [pos, setPos] = useState(INIT_POS);
  const [showKill, setShowKill] = useState(false);
  const [privacy, setPrivacy] = useState(false);
  const [time, setTime] = useState(new Date());
  useEffect(() => { const t = setInterval(() => setTime(new Date()), 1000); return () => clearInterval(t); }, []);

  const toggle = id => setStrats(s => s.map(st => st.id === id ? {...st, status: st.status === "running" ? "paused" : "running"} : st));
  const killPos = id => setPos(p => p.filter(x => x.id !== id));
  const killAll = () => { setPos([]); setStrats(s => s.map(st => ({...st, status:"stopped", deployed:0}))); setShowKill(false); };

  const tC = strats.reduce((a,s) => a+s.capital, 0);
  const tD = strats.reduce((a,s) => a+s.deployed, 0);
  const tPd = strats.reduce((a,s) => a+s.pnlToday, 0);
  const tPa = strats.reduce((a,s) => a+s.pnlTotal, 0);
  const run = strats.filter(s => s.status === "running").length;
  const dangerFlags = pos.filter(p => p.flags.length > 0);
  const stratErrors = strats.filter(s => s.status === "error");
  const capUtil = ((tD / tC) * 100);
  const hasAlerts = dangerFlags.length > 0 || stratErrors.length > 0 || capUtil > 75;

  return (
    <div className="min-h-screen" style={{background:"linear-gradient(160deg,#f8fafc 0%,#eef2f7 50%,#e4eaf1 100%)",fontFamily:"'DM Sans','Segoe UI',system-ui,sans-serif"}}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
        .mono{font-family:'JetBrains Mono',monospace}
        .tab-on{background:white;box-shadow:0 1px 3px rgba(0,0,0,.08)}
        .kill-btn{background:linear-gradient(135deg,#dc2626,#b91c1c)}.kill-btn:hover{background:linear-gradient(135deg,#b91c1c,#991b1b)}
        @keyframes fu{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}.fu{animation:fu .2s ease-out}
        .hovr:hover{background:#f8fafc}
        .rn{min-width:20px;height:20px;border-radius:5px;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;flex-shrink:0}
        .blur-it{filter:blur(8px);user-select:none;pointer-events:none}
        .arrow-btn{width:36px;height:36px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:16px;font-weight:700;border:1px solid #e2e8f0;background:white;cursor:pointer;transition:all .15s}
        .arrow-btn:hover{background:#f1f5f9;box-shadow:0 2px 4px rgba(0,0,0,.06)}.arrow-btn:disabled{opacity:.25;cursor:default}
        .flag-pulse{animation:pulse 2s infinite}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:.6}}
      `}</style>

      {/* HEADER */}
      <header className="bg-white border-b border-slate-200 px-4 py-2.5 flex items-center justify-between sticky top-0 z-50" style={{boxShadow:"0 1px 3px rgba(0,0,0,.03)"}}>
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center text-white font-bold text-xs" style={{background:"linear-gradient(135deg,#0f172a,#334155)"}}>SC</div>
          <h1 className="text-base font-bold text-slate-900 tracking-tight">Strategy Command Center</h1>
        </div>
        <div className="flex items-center gap-2.5">
          <span className="flex items-center gap-1.5 text-xs text-slate-500"><span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"/><span className="mono">{time.toLocaleTimeString("en-IN",{hour12:false})}</span></span>
          <button onClick={() => setPrivacy(!privacy)} className={`px-2.5 py-1 rounded-lg text-xs font-semibold border transition-all ${privacy?"bg-slate-900 text-white border-slate-900":"bg-white text-slate-500 border-slate-200"}`}>{privacy?"🔒":"🔓"}</button>
          {hasAlerts && <span className="w-2 h-2 rounded-full bg-red-500 flag-pulse"/>}
          <button onClick={() => setShowKill(true)} className="kill-btn text-white px-3 py-1 rounded-lg text-xs font-bold hover:scale-105 transition-transform">✕ KILL ALL</button>
        </div>
      </header>

      {/* Kill Modal */}
      {showKill && <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50" onClick={() => setShowKill(false)}>
        <div className="bg-white rounded-2xl p-7 max-w-sm w-full shadow-2xl fu" onClick={e => e.stopPropagation()}>
          <h3 className="text-lg font-bold text-slate-900 mb-2">⚠️ Emergency Kill All</h3>
          <div className="bg-red-50 rounded-xl p-3 mb-5 text-sm text-red-700">Close <strong>{pos.length}</strong> positions, stop <strong>{run}</strong> strategies. Cannot undo.</div>
          <div className="flex gap-3">
            <button onClick={() => setShowKill(false)} className="flex-1 px-4 py-2 rounded-xl border border-slate-200 text-slate-600 font-semibold text-sm">Cancel</button>
            <button onClick={killAll} className="flex-1 kill-btn text-white px-4 py-2 rounded-xl font-bold text-sm">Confirm</button>
          </div>
        </div>
      </div>}

      {/* INLINE ALERT BANNER — appears on every tab when there are flags */}
      {hasAlerts && <div className="mx-4 mt-3 bg-red-50 border border-red-200 rounded-xl px-4 py-2.5 flex items-start gap-3">
        <span className="text-base mt-0.5">🚨</span>
        <div className="flex-1 space-y-1">
          {stratErrors.map(s => <p key={s.id} className="text-xs text-red-700 font-medium">⚠️ <strong>{s.name}</strong>: Strategy in error state — Max DD breached ({s.maxDD}%)</p>)}
          {dangerFlags.map(p => p.flags.map((f,i) => <p key={p.id+i} className="text-xs text-red-700 font-medium">🔴 <strong>{p.symbol}</strong>: {f}</p>))}
          {capUtil > 75 && <p className="text-xs text-amber-700 font-medium">⚡ Capital utilization at <strong>{capUtil.toFixed(0)}%</strong> — consider reducing exposure</p>}
        </div>
        <button onClick={() => setTab("positions")} className="text-xs text-red-600 font-bold hover:underline whitespace-nowrap">View →</button>
      </div>}

      {/* TABS — only 4 */}
      <nav className="px-4 pt-3 pb-1 overflow-x-auto" style={{scrollbarWidth:"none"}}>
        <div className="inline-flex bg-slate-100 rounded-xl p-1 gap-0.5">
          {TABS.map(t => <button key={t.id} onClick={() => setTab(t.id)} className={`px-3.5 py-1.5 rounded-lg text-sm font-semibold transition-all whitespace-nowrap ${tab===t.id?"tab-on text-slate-900":"text-slate-500 hover:text-slate-700"}`}>{t.em} {t.label}</button>)}
        </div>
      </nav>

      <main className="px-4 pb-8 fu" key={tab}>
        {tab === "dash" && <DashTab s={strats} p={pos} tc={tC} td={tD} tp={tPd} ta={tPa} r={run} tgl={toggle} nav={setTab} priv={privacy} capUtil={capUtil}/>}
        {tab === "positions" && <PosTradesTab pos={pos} kp={killPos} priv={privacy} strats={strats}/>}
        {tab === "blueprints" && <BlueprintsTab priv={privacy}/>}
        {tab === "deepdive" && <DeepDiveTab priv={privacy}/>}
      </main>
    </div>
  );
}

// ═══════════════ DASHBOARD ═══════════════════════════════════
function DashTab({s, p, tc, td, tp, ta, r, tgl, nav, priv, capUtil}) {
  return <div className="space-y-4 pt-2">
    {/* Metrics row */}
    <div className={`grid grid-cols-3 lg:grid-cols-6 gap-2.5 ${priv?"blur-it":""}`}>
      <MC label="Today P&L" value={fmt(tp)} accent={cl(tp)} sub={pc(tp/tc*100)}/><MC label="Total P&L" value={fmt(ta)} accent={cl(ta)}/><MC label="Deployed" value={fmt(td)} accent="text-sky-700" sub={`${capUtil.toFixed(0)}% of ${fmt(tc)}`}/><MC label="Free Cash" value={fmt(tc-td)} accent="text-emerald-700"/><MC label="Strategies" value={`${r}/${s.length}`} sub={`${r} live`}/><MC label="Positions" value={p.length} sub="open"/>
    </div>

    {/* Capital utilization bar — inline, not a separate tab */}
    <div className={`bg-white rounded-xl border border-slate-200 p-4 shadow-sm ${priv?"blur-it":""}`}>
      <div className="flex items-center justify-between mb-2"><h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">Capital Deployment</h3><span className={`text-xs font-bold mono ${capUtil>75?"text-red-600":capUtil>60?"text-amber-600":"text-emerald-600"}`}>{capUtil.toFixed(0)}% used</span></div>
      <div className="h-2.5 bg-slate-100 rounded-full overflow-hidden mb-3"><div className="h-full rounded-full transition-all" style={{width:`${capUtil}%`,background:capUtil>80?"#dc2626":capUtil>60?"#f59e0b":"#059669"}}/></div>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">{s.map(st => {
        const u = st.capital > 0 ? (st.deployed/st.capital*100) : 0;
        return <div key={st.id} className="text-xs"><div className="flex justify-between mb-0.5"><span className="text-slate-600 truncate font-medium" style={{maxWidth:90}}>{st.name}</span><span className="mono text-slate-400">{u.toFixed(0)}%</span></div><div className="h-1 bg-slate-100 rounded-full"><div className="h-full rounded-full" style={{width:`${u}%`,background:u>85?"#dc2626":u>60?"#f59e0b":"#059669"}}/></div></div>;
      })}</div>
    </div>

    <div className="grid lg:grid-cols-3 gap-4">
      {/* Charts */}
      <div className={`bg-white rounded-xl border border-slate-200 p-4 shadow-sm ${priv?"blur-it":""}`}><h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">Weekly P&L</h3><MiniBar data={PNL_W} height={64}/><div className="mt-1.5 text-right"><span className={`mono text-xs font-bold ${cl(PNL_W.reduce((a,d)=>a+d.pnl,0))}`}>{fmt(PNL_W.reduce((a,d)=>a+d.pnl,0))}</span></div></div>
      <div className={`bg-white rounded-xl border border-slate-200 p-4 shadow-sm ${priv?"blur-it":""}`}><h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">Monthly (6M)</h3><MiniBar data={PNL_M} height={64}/><div className="mt-1.5 text-right"><span className={`mono text-xs font-bold ${cl(PNL_M.reduce((a,d)=>a+d.pnl,0))}`}>{fmt(PNL_M.reduce((a,d)=>a+d.pnl,0))}</span></div></div>

      {/* Strategy quick controls */}
      <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
        <div className="flex items-center justify-between mb-2"><h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">Strategies</h3></div>
        <div className="space-y-1.5">{s.map(st => <div key={st.id} className={`flex items-center justify-between py-1 px-2 rounded-lg ${st.status==="error"?"bg-red-50":""}`}><div className="flex items-center gap-2"><SB s={st.status}/><span className="text-xs font-medium text-slate-700 truncate" style={{maxWidth:100}}>{st.name}</span>{st.status==="error" && <span className="text-xs text-red-500 font-bold">⚠️</span>}</div><div className="flex items-center gap-1.5"><span className={`mono text-xs font-bold ${cl(st.pnlToday)}`}>{fmt(st.pnlToday)}</span><button onClick={() => tgl(st.id)} className={`w-6 h-6 rounded-md flex items-center justify-center text-xs ${st.status==="running"?"bg-emerald-50 text-emerald-600":"bg-slate-100 text-slate-400"}`}>{st.status==="running"?"⏸":"▶"}</button></div></div>)}</div>
      </div>
    </div>

    {/* Recent journal entries — inline, not separate tab */}
    <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
      <div className="flex items-center justify-between mb-2"><h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">Recent Journal Notes</h3><button onClick={() => nav("positions")} className="text-xs text-sky-600 font-semibold hover:underline">All trades →</button></div>
      <div className="space-y-2">{TRADE_HIST.filter(t => t.journal).slice(0,3).map(t => <div key={t.id} className="flex items-start gap-2 text-xs"><span className={`mono font-bold ${cl(t.pnl)} flex-shrink-0 w-16`}>{fmt(t.pnl)}</span><span className="text-slate-600 flex-1">{t.journal}</span><span className="text-slate-400 flex-shrink-0">{t.date}</span></div>)}</div>
    </div>
  </div>;
}

// ═══════════════ POSITIONS & TRADES (merged) ═════════════════
function PosTradesTab({pos, kp, priv, strats}) {
  const [cf, setCf] = useState(null);
  const [view, setView] = useState("live"); // live | history
  const [histFilter, setHistFilter] = useState("all");
  const [editJournal, setEditJournal] = useState(null);
  const [journalText, setJournalText] = useState("");
  const [journals, setJournals] = useState(() => {
    const m = {}; TRADE_HIST.forEach(t => { if (t.journal) m[t.id] = t.journal; }); return m;
  });

  const filtHist = histFilter === "all" ? TRADE_HIST : TRADE_HIST.filter(t => t.pnl > 0 ? histFilter === "wins" : histFilter === "losses");

  const saveJournal = (id) => { setJournals({...journals, [id]: journalText}); setEditJournal(null); };

  return <div className="pt-2 space-y-4">
    {/* Toggle live/history */}
    <div className="flex items-center justify-between">
      <div className="inline-flex bg-slate-100 rounded-lg p-0.5"><button onClick={() => setView("live")} className={`px-3.5 py-1.5 rounded-md text-sm font-semibold ${view==="live"?"tab-on text-slate-900":"text-slate-500"}`}>📊 Live ({pos.length})</button><button onClick={() => setView("history")} className={`px-3.5 py-1.5 rounded-md text-sm font-semibold ${view==="history"?"tab-on text-slate-900":"text-slate-500"}`}>📜 History</button></div>
      {view === "history" && <div className="inline-flex bg-slate-100 rounded-lg p-0.5 text-xs font-semibold">{["all","wins","losses"].map(f => <button key={f} onClick={() => setHistFilter(f)} className={`px-2.5 py-1 rounded-md capitalize ${histFilter===f?"bg-white text-slate-900 shadow-sm":"text-slate-500"}`}>{f}</button>)}</div>}
    </div>

    {view === "live" ? (
      <>
        {pos.length === 0 ? <div className="bg-white rounded-xl border p-10 text-center text-slate-400">No open positions</div> : (
          <div className={`bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm ${priv?"blur-it":""}`}>
            <div className="overflow-x-auto"><table className="w-full text-xs"><thead><tr className="bg-slate-50 border-b border-slate-200">{["","Strategy","Symbol","Side","Qty","Entry","Current","P&L","P&L%","SL","TP",""].map(h => <th key={h} className="px-2.5 py-2.5 text-left font-semibold text-slate-500 uppercase tracking-wider whitespace-nowrap">{h}</th>)}</tr></thead>
              <tbody>{pos.map(x => (
                <tr key={x.id} className={`border-b border-slate-100 hovr ${x.flags.length>0?"bg-red-50 bg-opacity-50":""}`}>
                  {/* Flag indicator */}
                  <td className="pl-2.5 py-2">{x.flags.length > 0 && <span className="text-red-500 flag-pulse text-sm">⚠️</span>}</td>
                  <td className="px-2.5 py-2 text-slate-500 whitespace-nowrap">{x.strategy}</td>
                  <td className="px-2.5 py-2 font-bold text-slate-900 mono whitespace-nowrap">
                    {x.symbol}
                    {/* Inline flags right under symbol */}
                    {x.flags.length > 0 && <div className="flex flex-wrap gap-1 mt-0.5">{x.flags.map((f,i) => <span key={i} className="text-xs bg-red-100 text-red-700 px-1.5 py-0.5 rounded font-semibold">{f}</span>)}</div>}
                  </td>
                  <td className={`px-2.5 py-2 font-bold ${x.side==="LONG"?"text-emerald-600":"text-red-600"}`}>{x.side}</td>
                  <td className="px-2.5 py-2 mono">{x.qty}</td>
                  <td className="px-2.5 py-2 mono">{x.entry.toFixed(2)}</td>
                  <td className="px-2.5 py-2 mono font-semibold">{x.current.toFixed(2)}</td>
                  <td className={`px-2.5 py-2 mono font-bold ${cl(x.pnl)}`}>{fmt(x.pnl)}</td>
                  <td className={`px-2.5 py-2 mono font-bold ${cl(x.pnlPct)}`}>{pc(x.pnlPct)}</td>
                  <td className="px-2.5 py-2 mono text-red-400">{x.sl}</td>
                  <td className="px-2.5 py-2 mono text-emerald-500">{x.tp||"—"}</td>
                  <td className="px-2.5 py-2">{cf===x.id ? <div className="flex gap-1"><button onClick={() => {kp(x.id);setCf(null)}} className="kill-btn text-white px-2 py-0.5 rounded font-bold">Exit</button><button onClick={() => setCf(null)} className="bg-slate-100 text-slate-500 px-2 py-0.5 rounded">No</button></div> : <button onClick={() => setCf(x.id)} className="bg-red-50 text-red-600 px-2 py-0.5 rounded font-semibold hover:bg-red-100 whitespace-nowrap">Close</button>}</td>
                </tr>
              ))}</tbody>
            </table></div>
          </div>
        )}

        {/* Risk summary strip — contextual, not separate tab */}
        <div className={`grid grid-cols-3 sm:grid-cols-6 gap-2 ${priv?"blur-it":""}`}>
          {strats.map(st => <div key={st.id} className={`bg-white rounded-lg border p-2.5 text-center ${st.status==="error"?"border-red-200 bg-red-50":"border-slate-200"}`}><div className="text-xs text-slate-500 truncate">{st.name}</div><div className={`mono text-sm font-bold ${cl(st.pnlToday)}`}>{fmt(st.pnlToday)}</div><div className="flex justify-center gap-1 mt-1"><RB r={st.risk}/><span className="mono text-xs text-slate-400">{st.sharpe}S</span></div></div>)}
        </div>
      </>
    ) : (
      /* TRADE HISTORY with inline journal */
      <div className="space-y-2.5">
        {filtHist.map(t => (
          <div key={t.id} className={`bg-white rounded-xl border shadow-sm ${t.pnl>0?"border-emerald-100":"border-red-100"}`}>
            <div className="p-4">
              <div className="flex flex-wrap items-start justify-between gap-2 mb-1.5">
                <div><div className="flex items-center gap-2 mb-0.5"><span className="font-bold text-slate-900 mono text-sm">{t.symbol}</span><span className={`text-xs font-bold ${t.side==="LONG"?"text-emerald-600":"text-red-600"}`}>{t.side}</span><span className="text-xs bg-slate-100 text-slate-500 px-1.5 py-0.5 rounded">{t.strategy}</span></div><span className="text-xs text-slate-400">{t.date} · {t.holding}</span></div>
                <div className="text-right"><div className={`text-lg font-bold mono ${cl(t.pnl)}`}>{fmt(t.pnl)}</div><div className={`text-xs mono font-semibold ${cl(t.pnlPct)}`}>{pc(t.pnlPct)}</div></div>
              </div>
              {t.notes && <p className="text-xs text-slate-500 mb-2">📋 {t.notes}</p>}

              {/* Inline journal — right here in the trade card */}
              {editJournal === t.id ? (
                <div className="bg-amber-50 rounded-lg p-3 space-y-2">
                  <textarea rows={2} value={journalText} onChange={e => setJournalText(e.target.value)} placeholder="Your thoughts, lessons, what you'd do differently..." className="w-full px-3 py-2 rounded-lg border border-amber-200 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-amber-300 bg-white"/>
                  <div className="flex gap-2"><button onClick={() => saveJournal(t.id)} className="bg-slate-900 text-white px-3 py-1 rounded-lg text-xs font-semibold">Save</button><button onClick={() => setEditJournal(null)} className="text-slate-500 text-xs font-semibold">Cancel</button></div>
                </div>
              ) : journals[t.id] ? (
                <div className="bg-amber-50 rounded-lg px-3 py-2 flex items-start gap-2 cursor-pointer hover:bg-amber-100 transition-colors" onClick={() => { setEditJournal(t.id); setJournalText(journals[t.id]); }}>
                  <span className="text-xs">📝</span>
                  <span className="text-xs text-amber-800 flex-1">{journals[t.id]}</span>
                  <span className="text-xs text-amber-500">edit</span>
                </div>
              ) : (
                <button onClick={() => { setEditJournal(t.id); setJournalText(""); }} className="text-xs text-slate-400 hover:text-slate-600 font-medium">+ Add journal note</button>
              )}
            </div>
          </div>
        ))}
      </div>
    )}
  </div>;
}

// ═══════════════ BLUEPRINTS ══════════════════════════════════
function BlueprintsTab({priv}) {
  const [sel, setSel] = useState(BLUEPRINTS[0].id);
  const [view, setView] = useState("rules");
  const bp = BLUEPRINTS.find(b => b.id === sel);
  const bt = bp?.backtest;

  const RL = ({rules, color}) => <div className="space-y-2">{rules.map((r,i) => <div key={i} className="flex items-start gap-2"><span className={`rn ${color}`}>{i+1}</span><span className="text-sm text-slate-700 leading-relaxed">{r}</span></div>)}</div>;
  const BM = ({label, value, warn}) => <div><span className="text-xs text-slate-400">{label}</span><span className={`text-sm font-bold mono block ${warn?"text-red-600":"text-slate-800"}`}>{value}</span></div>;

  return <div className="pt-2 space-y-4">
    <div className="flex flex-wrap gap-2 overflow-x-auto pb-1" style={{scrollbarWidth:"none"}}>{BLUEPRINTS.map(b => {
      const st = INIT_STRATS.find(s => s.id === b.id);
      return <button key={b.id} onClick={() => setSel(b.id)} className={`flex items-center gap-2 px-3 py-2 rounded-xl text-sm font-semibold whitespace-nowrap border transition-all ${sel===b.id?"bg-white border-slate-300 text-slate-900 shadow-sm":"border-transparent text-slate-500 hover:bg-white hover:border-slate-200"}`}>{st && <SB s={st.status}/>}{b.name}</button>;
    })}</div>

    {bp && <div className="fu" key={bp.id+view}>
      <div className="bg-white rounded-t-2xl border border-slate-200 border-b-0 p-5">
        <div className="flex flex-wrap items-start justify-between gap-3 mb-2">
          <div><div className="flex flex-wrap items-center gap-2 mb-1"><h3 className="text-lg font-bold text-slate-900">{bp.name}</h3><span className="mono text-xs bg-slate-100 text-slate-500 px-2 py-0.5 rounded-lg font-semibold">{bp.version}</span><SyB t={bp.systemType}/></div><p className="text-sm text-slate-600 max-w-xl">{bp.description}</p></div>
          <a href={bp.github} target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 bg-slate-900 text-white px-3 py-2 rounded-xl text-xs font-semibold hover:bg-slate-800 flex-shrink-0"><svg width="14" height="14" viewBox="0 0 24 24" fill="white"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>GitHub</a>
        </div>
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-400 mb-3"><span>⏱ <span className="text-slate-700 font-semibold">{bp.holdingPeriod}</span></span><span>📊 <span className="text-slate-700 font-semibold">{bp.timeframe}</span></span><span>🎯 <span className="text-slate-700 font-semibold">{bp.instruments}</span></span><span>💰 <span className="text-slate-700 font-semibold">{bp.capital}</span></span></div>
        <div className="text-xs text-slate-400 mb-3">Commit: <span className="mono text-slate-500">{bp.lastCommit}</span> — <span className="italic">{bp.commitMsg}</span></div>
        <div className="flex gap-1"><button onClick={() => setView("rules")} className={`px-3 py-1.5 rounded-lg text-xs font-semibold ${view==="rules"?"bg-slate-900 text-white":"bg-slate-100 text-slate-500"}`}>📋 Rules</button><button onClick={() => setView("backtest")} className={`px-3 py-1.5 rounded-lg text-xs font-semibold ${view==="backtest"?"bg-slate-900 text-white":"bg-slate-100 text-slate-500"}`}>📊 Backtest</button></div>
      </div>

      {view === "rules" ? (
        <div className={priv ? "blur-it" : ""}>
          <div className="grid lg:grid-cols-3 border border-slate-200 border-t-0 bg-white">
            <div className="p-5 lg:border-r border-slate-100"><div className="flex items-center gap-2 mb-3"><span className="w-7 h-7 rounded-lg bg-emerald-100 flex items-center justify-center text-sm">📥</span><div><h4 className="text-sm font-bold text-slate-900">Entry</h4><span className="text-xs text-slate-400">ALL must be true</span></div></div><RL rules={bp.entry} color="bg-emerald-50 text-emerald-700"/></div>
            <div className="p-5 lg:border-r border-slate-100 border-t lg:border-t-0"><div className="flex items-center gap-2 mb-3"><span className="w-7 h-7 rounded-lg bg-sky-100 flex items-center justify-center text-sm">📤</span><div><h4 className="text-sm font-bold text-slate-900">Exit</h4><span className="text-xs text-slate-400">ANY triggers</span></div></div><RL rules={bp.exit} color="bg-sky-50 text-sky-700"/></div>
            <div className="p-5 border-t lg:border-t-0"><div className="flex items-center gap-2 mb-3"><span className="w-7 h-7 rounded-lg bg-red-100 flex items-center justify-center text-sm">🛑</span><div><h4 className="text-sm font-bold text-slate-900">Stop Loss</h4><span className="text-xs text-slate-400">{bp.stopLoss.length} layers</span></div></div><RL rules={bp.stopLoss} color="bg-red-50 text-red-700"/></div>
          </div>
          <div className="bg-white border border-slate-200 border-t-0 px-5 py-3"><p className="text-sm text-slate-700 bg-amber-50 rounded-lg px-3 py-2 font-medium">⚖️ {bp.positionSizing}</p></div>
          <div className="grid sm:grid-cols-2 bg-white border border-slate-200 border-t-0">
            <div className="px-5 py-3 sm:border-r border-slate-100"><span className="text-xs font-bold text-slate-400 uppercase">Indicators</span><div className="flex flex-wrap gap-1 mt-1">{bp.indicators.map((x,i) => <span key={i} className="mono text-xs bg-slate-100 text-slate-700 px-2 py-0.5 rounded">{x}</span>)}</div></div>
            <div className="px-5 py-3 border-t sm:border-t-0"><span className="text-xs font-bold text-slate-400 uppercase">Filters</span><div className="flex flex-wrap gap-1 mt-1">{bp.filters.map((x,i) => <span key={i} className="text-xs bg-slate-50 text-slate-600 px-2 py-0.5 rounded border border-slate-200">{x}</span>)}</div></div>
          </div>
          <div className="bg-white rounded-b-2xl border border-slate-200 border-t-0 px-5 py-2.5 flex items-center gap-2"><span className="text-xs text-slate-400 font-semibold">Tags:</span>{bp.tags.map((t,i) => <span key={i} className="text-xs bg-violet-50 text-violet-600 px-2 py-0.5 rounded-full font-semibold">#{t}</span>)}</div>
        </div>
      ) : (
        <div className="bg-white rounded-b-2xl border border-slate-200 border-t-0 p-5">
          <div className="flex items-center justify-between mb-3"><h4 className="text-sm font-bold text-slate-900">Backtest Results</h4><span className="text-xs bg-slate-100 text-slate-500 px-2 py-0.5 rounded-lg font-semibold mono">{bt.period}</span></div>
          <div className="grid grid-cols-4 lg:grid-cols-8 gap-3 mb-4 pb-4 border-b border-slate-100"><BM label="Trades" value={bt.totalTrades}/><BM label="Win%" value={bt.winRate+"%"}/><BM label="PF" value={bt.profitFactor}/><BM label="Expectancy" value={(bt.expectancy>0?"+":"")+bt.expectancy+"R"} warn={bt.expectancy<0}/><BM label="Sharpe" value={bt.sharpe}/><BM label="Sortino" value={bt.sortino}/><BM label="Calmar" value={bt.calmar}/><BM label="Avg Hold" value={bt.avgHolding}/></div>
          <div className="bg-red-50 rounded-xl p-4 mb-4"><h5 className="text-xs font-bold text-red-800 uppercase tracking-wider mb-2">📉 Drawdown — What to Expect</h5><div className="grid grid-cols-2 sm:grid-cols-4 gap-3"><div><span className="text-xs text-red-500">Max Strategy DD</span><span className="text-lg font-bold mono text-red-700 block">{bt.maxDD}%</span></div><div><span className="text-xs text-red-500">Avg Strategy DD</span><span className="text-lg font-bold mono text-red-700 block">{bt.avgDD}%</span></div><div><span className="text-xs text-red-500">Max DD/Trade</span><span className="text-lg font-bold mono text-red-700 block">{bt.maxDDTrade}%</span></div><div><span className="text-xs text-red-500">Avg DD/Trade</span><span className="text-lg font-bold mono text-red-700 block">{bt.avgDDTrade}%</span></div></div><div className="mt-2 pt-2 border-t border-red-200 text-xs text-red-600">Max consecutive losses: <strong className="mono">{bt.maxConsecLoss}</strong> · Avg loss: <strong className="mono">{bt.avgLoss}%</strong></div></div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-3"><BM label="Avg Win" value={"+"+bt.avgWin+"%"}/><BM label="Avg Loss" value={bt.avgLoss+"%"} warn/><BM label="Best Month" value={"+"+bt.bestMonth+"%"}/><BM label="Worst Month" value={bt.worstMonth+"%"} warn/></div>
          <div className="bg-emerald-50 rounded-lg px-3 py-2 text-sm"><span className="font-semibold text-emerald-800">Monthly:</span><span className="mono font-bold text-emerald-700 ml-1">{bt.avgMonthly>0?"+":""}{bt.avgMonthly}%</span><span className="text-emerald-600 ml-1 text-xs">avg over {bt.period}</span></div>
        </div>
      )}
    </div>}
  </div>;
}

// ═══════════════ DAY DEEP DIVE ═══════════════════════════════
function DeepDiveTab({priv}) {
  const [idx, setIdx] = useState(0);
  const day = DAY_DATA[idx];
  const canPrev = idx < DAY_DATA.length - 1;
  const canNext = idx > 0;

  useEffect(() => {
    const h = e => { if (e.key === "ArrowLeft" && canPrev) setIdx(i => i+1); if (e.key === "ArrowRight" && canNext) setIdx(i => i-1); };
    window.addEventListener("keydown", h); return () => window.removeEventListener("keydown", h);
  }, [canPrev, canNext]);

  const wr = day.trades > 0 ? (day.wins / day.trades * 100).toFixed(0) : 0;
  const stBreak = [...new Set(day.items.map(t => t.str))].map(s => ({name:s, pnl:day.items.filter(t => t.str === s).reduce((a,t) => a+t.pnl,0), cnt:day.items.filter(t => t.str === s).length}));

  return <div className="pt-2 space-y-4">
    <div className="flex items-center justify-between">
      <h2 className="text-base font-bold text-slate-900">Day Deep Dive</h2>
      <div className="flex items-center gap-2.5">
        <button className="arrow-btn" disabled={!canPrev} onClick={() => setIdx(i => i+1)}>←</button>
        <div className="text-center min-w-[120px]"><div className="font-bold text-slate-900 text-sm">{day.date}</div><div className="text-xs text-slate-400">{day.dow} · {idx+1}/{DAY_DATA.length}</div></div>
        <button className="arrow-btn" disabled={!canNext} onClick={() => setIdx(i => i-1)}>→</button>
      </div>
    </div>

    <div className={`fu ${priv?"blur-it":""}`} key={idx}>
      <div className="grid grid-cols-3 sm:grid-cols-6 gap-2.5 mb-4">
        <MC label="Day P&L" value={fmt(day.pnl)} accent={cl(day.pnl)}/><MC label="Trades" value={day.trades} sub={`${day.wins}W/${day.losses}L`}/><MC label="Win Rate" value={wr+"%"} accent={Number(wr)>=50?"text-emerald-700":"text-red-600"}/><MC label="Best" value={fmt(day.best)} accent="text-emerald-700"/><MC label="Worst" value={fmt(day.worst)} accent="text-red-600"/><MC label="Capital" value={fmt(day.capUsed)}/>
      </div>

      {/* Strategy P&L breakdown — compact inline */}
      <div className="flex flex-wrap gap-2 mb-4">{stBreak.map(s => <div key={s.name} className="bg-white rounded-lg border border-slate-200 px-3 py-2 flex items-center gap-2"><span className="text-xs text-slate-600 font-medium">{s.name}</span><span className={`mono text-xs font-bold ${cl(s.pnl)}`}>{fmt(s.pnl)}</span><span className="text-xs text-slate-400">{s.cnt}t</span></div>)}</div>

      {/* Trades */}
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm">
        <div className="px-4 py-2.5 border-b border-slate-100 flex justify-between"><span className="text-sm font-semibold text-slate-700">All Trades</span><span className={`mono text-sm font-bold ${cl(day.pnl)}`}>Net: {fmt(day.pnl)}</span></div>
        <div className="divide-y divide-slate-100">{day.items.map((t,i) => (
          <div key={i} className="px-4 py-2.5 hovr">
            <div className="flex items-center justify-between mb-0.5">
              <div className="flex items-center gap-2"><span className="font-bold text-slate-900 mono text-xs">{t.sym}</span><span className={`text-xs font-bold ${t.side==="LONG"?"text-emerald-600":"text-red-600"}`}>{t.side}</span><span className="text-xs bg-slate-100 text-slate-500 px-1.5 py-0.5 rounded">{t.str}</span></div>
              <div className="flex items-center gap-2"><span className={`mono text-xs font-bold ${cl(t.pnl)}`}>{fmt(t.pnl)}</span><span className={`mono text-xs ${cl(t.pp)}`}>{pc(t.pp)}</span></div>
            </div>
            <div className="flex gap-3 text-xs text-slate-400"><span>En: <span className="mono text-slate-600">{t.en}</span></span><span>Ex: <span className="mono text-slate-600">{t.ex}</span></span><span>⏱ {t.time}</span></div>
            {t.note && <p className="text-xs text-slate-500 mt-1">📝 {t.note}</p>}
          </div>
        ))}</div>
      </div>
    </div>
    <p className="text-xs text-slate-400 text-center">← → keys or buttons to navigate days</p>
  </div>;
}
