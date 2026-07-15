"""Canonical Indian-market cost model for research/81 (brief section 3.2).

All costs are PER SIDE unless stated. Conservative Zerodha-style retail rates
(2025-26). Slippage is a parameter so every backtest can run 1x and 2x.

Products:
  - FUTURES_PROXY: user decision 2026-07-15 — cash/index series traded "as"
    futures (lot-sized, marginable, shortable). Futures charges apply.
  - CASH_INTRADAY: equity MIS, squared off same day.
  - CASH_DELIVERY: equity CNC, held overnight (the usual swing case).

Numbers (verify before live deploy; fine for research):
  Futures : brokerage min(0.03%, Rs20)/order, STT 0.02% sell, exch 0.00173%,
            SEBI 0.0001%, stamp 0.002% buy, GST 18% on (brokerage+exch+sebi)
  Intraday: brokerage min(0.03%, Rs20)/order, STT 0.025% sell, exch 0.00297%,
            SEBI 0.0001%, stamp 0.003% buy, GST 18%
  Delivery: brokerage 0, STT 0.1% BOTH sides, exch 0.00297%, SEBI 0.0001%,
            stamp 0.015% buy, DP Rs15.34 per sell scrip-day, GST on exch+sebi
"""
from __future__ import annotations

from dataclasses import dataclass

GST = 0.18

# default slippage assumptions (fraction of price, per side)
SLIPPAGE_LIQUID = 0.0003     # 3 bps — NIFTY/BANKNIFTY/F&O large caps
SLIPPAGE_MID = 0.0007        # 7 bps — liquid cash names outside F&O
SLIPPAGE_ILLIQUID = 0.0015   # 15 bps — bottom-of-universe cash


@dataclass(frozen=True)
class CostConfig:
    product: str = "FUTURES_PROXY"      # FUTURES_PROXY | CASH_INTRADAY | CASH_DELIVERY
    slippage: float = SLIPPAGE_LIQUID   # per-side, fraction of price
    slippage_mult: float = 1.0          # run every candidate at 1.0 and 2.0

    def side_cost(self, price: float, qty: float, is_buy: bool) -> float:
        """Rupee cost of one side (excl. slippage — slippage adjusts fill price)."""
        turnover = price * qty
        if self.product == "FUTURES_PROXY":
            brok = min(0.0003 * turnover, 20.0)
            stt = 0.0 if is_buy else 0.0002 * turnover
            exch = 0.0000173 * turnover
            sebi = 0.000001 * turnover
            stamp = 0.00002 * turnover if is_buy else 0.0
            gst = GST * (brok + exch + sebi)
        elif self.product == "CASH_INTRADAY":
            brok = min(0.0003 * turnover, 20.0)
            stt = 0.0 if is_buy else 0.00025 * turnover
            exch = 0.0000297 * turnover
            sebi = 0.000001 * turnover
            stamp = 0.00003 * turnover if is_buy else 0.0
            gst = GST * (brok + exch + sebi)
        elif self.product == "CASH_DELIVERY":
            brok = 0.0
            stt = 0.001 * turnover                  # both sides
            exch = 0.0000297 * turnover
            sebi = 0.000001 * turnover
            stamp = 0.00015 * turnover if is_buy else 0.0
            dp = 0.0 if is_buy else 15.34
            gst = GST * (exch + sebi)
            return brok + stt + exch + sebi + stamp + dp + gst
        else:
            raise ValueError(f"unknown product {self.product}")
        return brok + stt + exch + sebi + stamp + gst

    def fill_price(self, price: float, is_buy: bool) -> float:
        """Fill adjusted for slippage (buys fill higher, sells lower)."""
        slip = self.slippage * self.slippage_mult
        return price * (1 + slip) if is_buy else price * (1 - slip)

    def round_trip_bps(self, price: float, qty: float) -> float:
        """Total round-trip cost incl. slippage, in bps of one-side turnover.
        The brief's expectancy gate: per-trade edge must exceed 2x this."""
        turnover = price * qty
        charges = self.side_cost(price, qty, True) + self.side_cost(price, qty, False)
        slip = 2 * self.slippage * self.slippage_mult * turnover
        return (charges + slip) / turnover * 1e4
