"""
Indian intraday charges calculator (Zerodha).

Reference: https://zerodha.com/charges (NSE equity intraday + F&O)

Equity intraday (MIS):
  brokerage  = min(Rs 20, 0.03% of turnover) per executed order
  STT        = 0.025% on sell side (turnover_sell)
  exchange   = 0.00322% NSE on (turnover_buy + turnover_sell)
  GST        = 18% on (brokerage + exchange + sebi)
  SEBI       = Rs 10 per crore on (turnover_buy + turnover_sell)
  stamp      = 0.003% on buy side (turnover_buy)

Equity delivery (CNC):
  brokerage  = 0
  STT        = 0.1% on buy AND sell side (turnover_buy + turnover_sell)
  exchange   = 0.00322% NSE
  GST        = 18% on (exchange + sebi)
  SEBI       = Rs 10 per crore
  stamp      = 0.015% on buy side

Options (intraday or expiry):
  brokerage  = Rs 20 flat per executed order (no percentage cap)
  STT        = 0.1% on premium (sell side) for options
  exchange   = 0.0503% NSE on premium turnover
  GST        = 18% on (brokerage + exchange + sebi)
  SEBI       = Rs 10 per crore
  stamp      = 0.003% on buy premium

Futures (intraday):
  brokerage  = min(Rs 20, 0.03% of turnover) per executed order
  STT        = 0.02% on sell side
  exchange   = 0.00188% NSE
  GST        = 18% on (brokerage + exchange + sebi)
  SEBI       = Rs 10 per crore
  stamp      = 0.002% on buy side

This module returns total round-trip charges for one trade (buy + sell).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


InstrumentType = Literal['EQUITY', 'INDEX', 'OPTION', 'FUTURE']
ProductType = Literal['MIS', 'CNC', 'NRML']


@dataclass
class ChargeBreakdown:
    brokerage: float
    stt: float
    exchange_tx: float
    sebi: float
    stamp: float
    gst: float
    total: float

    def as_dict(self):
        return {
            'brokerage': round(self.brokerage, 2),
            'stt': round(self.stt, 2),
            'exchange_tx': round(self.exchange_tx, 2),
            'sebi': round(self.sebi, 2),
            'stamp': round(self.stamp, 2),
            'gst': round(self.gst, 2),
            'total': round(self.total, 2),
        }


def compute_charges(
    instrument_type: InstrumentType,
    direction: str,           # 'LONG' or 'SHORT'
    qty: int,
    entry_price: float,
    exit_price: float,
    product: ProductType = 'MIS',
) -> ChargeBreakdown:
    """Round-trip charges for one closed trade.

    Returns ChargeBreakdown. Always positive (subtract from gross to get net).
    """
    if qty is None or entry_price is None or exit_price is None:
        return ChargeBreakdown(0, 0, 0, 0, 0, 0, 0)

    qty = abs(int(qty))
    if qty == 0:
        return ChargeBreakdown(0, 0, 0, 0, 0, 0, 0)

    # In LONG, buy at entry, sell at exit. In SHORT, sell at entry, buy at exit.
    if direction == 'SHORT':
        buy_price, sell_price = exit_price, entry_price
    else:
        buy_price, sell_price = entry_price, exit_price

    turnover_buy = buy_price * qty
    turnover_sell = sell_price * qty
    turnover_total = turnover_buy + turnover_sell

    inst = (instrument_type or 'EQUITY').upper()

    if inst == 'OPTION':
        # Two executed orders
        brokerage = 20.0 + 20.0
        stt = 0.001 * turnover_sell
        exchange_tx = 0.000503 * turnover_total
        sebi = 10.0 / 1e7 * turnover_total
        stamp = 0.00003 * turnover_buy
    elif inst == 'FUTURE':
        brokerage = min(20.0, 0.0003 * turnover_buy) + min(20.0, 0.0003 * turnover_sell)
        stt = 0.0002 * turnover_sell
        exchange_tx = 0.0000188 * turnover_total
        sebi = 10.0 / 1e7 * turnover_total
        stamp = 0.00002 * turnover_buy
    elif inst == 'INDEX':
        # Indices aren't tradable directly — treat as zero charges (positions
        # that show up here are normally option/futures; this branch is
        # defensive)
        return ChargeBreakdown(0, 0, 0, 0, 0, 0, 0)
    else:
        # EQUITY
        if product == 'CNC':
            brokerage = 0.0
            stt = 0.001 * turnover_total  # 0.1% buy+sell
            exchange_tx = 0.0000322 * turnover_total
            sebi = 10.0 / 1e7 * turnover_total
            stamp = 0.00015 * turnover_buy
        else:
            # MIS intraday
            brokerage = min(20.0, 0.0003 * turnover_buy) + min(20.0, 0.0003 * turnover_sell)
            stt = 0.00025 * turnover_sell
            exchange_tx = 0.0000322 * turnover_total
            sebi = 10.0 / 1e7 * turnover_total
            stamp = 0.00003 * turnover_buy

    gst = 0.18 * (brokerage + exchange_tx + sebi)
    total = brokerage + stt + exchange_tx + sebi + stamp + gst
    return ChargeBreakdown(brokerage, stt, exchange_tx, sebi, stamp, gst, total)
