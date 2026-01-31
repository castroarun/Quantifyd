"""
Momentum + Quality Portfolio Management
========================================

Core portfolio management for the MQ strategy backtest engine.

Contains:
- MQBacktestConfig: All configurable parameters
- Position: Individual stock position with topup tracking
- Trade: Completed trade record (entry + exit)
- TopupRecord: Breakout topup event
- Portfolio: Full portfolio state with 80/20 allocation
- TransactionCostModel: Indian market costs
- Exit logic: Fundamental, ATH drawdown, hard stop, rebalance
- Sector concentration enforcement
"""

import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class ExitReason(str, Enum):
    FUNDAMENTAL_3Q = 'fundamental_3q_decline'
    FUNDAMENTAL_2Y = 'fundamental_2y_decline'
    ATH_DRAWDOWN = 'ath_drawdown_rebalance'
    HARD_STOP = 'hard_stop_loss'
    REBALANCE_REPLACE = 'rebalance_replaced'
    MANUAL = 'manual'


class SignalType(str, Enum):
    ENTRY = 'ENTRY'
    EXIT = 'EXIT'
    TOPUP = 'TOPUP'
    WARNING = 'WARNING'


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MQBacktestConfig:
    """All configurable parameters for the MQ strategy backtest."""

    # Time period
    start_date: str = '2023-01-01'
    end_date: str = '2025-12-31'

    # Capital
    initial_capital: float = 10_000_000  # ₹1 Crore
    equity_allocation_pct: float = 0.80
    debt_reserve_pct: float = 0.20
    debt_fund_annual_return: float = 0.065  # ~6.5% p.a.

    # Portfolio construction
    portfolio_size: int = 30
    max_position_size: float = 0.10  # 10% of portfolio
    max_sector_weight: float = 0.25  # 25% per sector
    max_stocks_per_sector: int = 6

    # Momentum filter
    ath_proximity_threshold: float = 0.10  # Within 10% of 52-week high

    # Fundamental quality (non-financial)
    min_revenue_growth_3y_cagr: float = 0.15
    require_revenue_positive_each_year: bool = True
    max_debt_to_equity: float = 0.20
    min_opm_3y: float = 0.15
    require_opm_no_decline: bool = True

    # Fundamental quality (financial)
    min_roa: float = 0.01
    min_roe: float = 0.12

    # Consolidation & breakout (topup)
    consolidation_days: int = 20
    consolidation_range_pct: float = 0.05
    breakout_volume_multiplier: float = 1.5
    topup_pct_of_initial: float = 0.20  # 20% of original entry
    topup_cooldown_days: int = 5

    # Exit rules
    rebalance_ath_drawdown: float = 0.20  # >20% from ATH → exit at rebalance
    quarterly_decline_threshold: float = 0.10  # >10% YoY decline
    quarterly_decline_count: int = 3  # 3 consecutive quarters → exit
    hard_stop_loss: float = 0.30  # 30% loss from entry

    # Rebalance schedule
    rebalance_months: List[int] = field(default_factory=lambda: [1, 7])

    # Transaction costs (Indian market)
    brokerage_pct: float = 0.0003  # 0.03% (Zerodha)
    stt_pct: float = 0.001  # 0.1% STT on sell side
    gst_pct: float = 0.18  # 18% GST on brokerage
    stamp_duty_pct: float = 0.00015  # 0.015%
    slippage_pct: float = 0.001  # 0.1%

    # Composite ranking weights
    weight_revenue: float = 0.30
    weight_debt: float = 0.25
    weight_opm: float = 0.25
    weight_opm_growth: float = 0.20

    @property
    def equity_capital(self) -> float:
        return self.initial_capital * self.equity_allocation_pct

    @property
    def debt_reserve_capital(self) -> float:
        return self.initial_capital * self.debt_reserve_pct

    @property
    def position_capital(self) -> float:
        """Capital per position at initial entry."""
        return self.equity_capital / self.portfolio_size


# =============================================================================
# Position & Trade Records
# =============================================================================

@dataclass
class Position:
    """An active stock position in the portfolio."""
    symbol: str
    sector: str
    entry_date: datetime
    entry_price: float
    shares: int
    initial_investment: float  # Original capital deployed (for topup calc)

    # Tracking
    current_price: float = 0.0
    rolling_ath: float = 0.0  # Highest price since entry
    topup_count: int = 0
    total_invested: float = 0.0  # initial + all topups
    total_shares: int = 0
    last_topup_date: Optional[datetime] = None

    # Fundamental monitoring
    quarterly_decline_streak: int = 0  # Consecutive quarters with >10% YoY decline

    def __post_init__(self):
        if self.current_price == 0:
            self.current_price = self.entry_price
        if self.rolling_ath == 0:
            self.rolling_ath = self.entry_price
        if self.total_invested == 0:
            self.total_invested = self.initial_investment
        if self.total_shares == 0:
            self.total_shares = self.shares

    @property
    def current_value(self) -> float:
        return self.total_shares * self.current_price

    @property
    def avg_price(self) -> float:
        return self.total_invested / self.total_shares if self.total_shares > 0 else 0

    @property
    def unrealized_pnl(self) -> float:
        return self.current_value - self.total_invested

    @property
    def unrealized_return_pct(self) -> float:
        return self.unrealized_pnl / self.total_invested if self.total_invested > 0 else 0

    @property
    def drawdown_from_ath(self) -> float:
        """Current drawdown from rolling ATH (0.0 = at ATH, 0.20 = 20% below)."""
        if self.rolling_ath <= 0:
            return 0.0
        return max(0, (self.rolling_ath - self.current_price) / self.rolling_ath)

    @property
    def loss_from_entry(self) -> float:
        """Loss from average entry price (0.0 = breakeven, 0.30 = 30% loss)."""
        if self.avg_price <= 0:
            return 0.0
        return max(0, (self.avg_price - self.current_price) / self.avg_price)

    def update_price(self, price: float):
        """Update current price and rolling ATH."""
        self.current_price = price
        if price > self.rolling_ath:
            self.rolling_ath = price

    def add_topup(self, shares: int, price: float, amount: float, date: datetime):
        """Record a topup purchase."""
        self.total_shares += shares
        self.total_invested += amount
        self.topup_count += 1
        self.last_topup_date = date
        self.current_price = price


@dataclass
class Trade:
    """Completed trade record (entry to exit)."""
    symbol: str
    sector: str

    # Entry
    entry_date: datetime
    entry_price: float
    shares_entered: int
    initial_investment: float

    # Exit
    exit_date: datetime
    exit_price: float
    exit_reason: ExitReason
    exit_value: float

    # Topups during hold
    topup_count: int = 0
    total_invested: float = 0.0  # initial + topups
    total_shares_at_exit: int = 0

    # P&L
    gross_pnl: float = 0.0
    transaction_costs: float = 0.0
    net_pnl: float = 0.0
    return_pct: float = 0.0
    holding_days: int = 0

    def __post_init__(self):
        if self.total_invested == 0:
            self.total_invested = self.initial_investment
        if self.total_shares_at_exit == 0:
            self.total_shares_at_exit = self.shares_entered
        if self.holding_days == 0:
            self.holding_days = (self.exit_date - self.entry_date).days


@dataclass
class TopupRecord:
    """Record of a breakout topup event."""
    symbol: str
    date: datetime
    price: float
    shares: int
    amount: float
    debt_fund_before: float
    debt_fund_after: float
    position_value_after: float
    topup_number: int  # 1st, 2nd, 3rd topup for this position


# =============================================================================
# Transaction Cost Model
# =============================================================================

class TransactionCostModel:
    """Indian equity market transaction costs."""

    def __init__(self, config: MQBacktestConfig):
        self.brokerage_pct = config.brokerage_pct
        self.stt_pct = config.stt_pct
        self.gst_pct = config.gst_pct
        self.stamp_duty_pct = config.stamp_duty_pct
        self.slippage_pct = config.slippage_pct

    def buy_cost(self, amount: float) -> float:
        """Total cost of buying (brokerage + GST on brokerage + stamp duty + slippage)."""
        brokerage = amount * self.brokerage_pct
        gst = brokerage * self.gst_pct
        stamp = amount * self.stamp_duty_pct
        slippage = amount * self.slippage_pct
        return brokerage + gst + stamp + slippage

    def sell_cost(self, amount: float) -> float:
        """Total cost of selling (brokerage + GST + STT + slippage)."""
        brokerage = amount * self.brokerage_pct
        gst = brokerage * self.gst_pct
        stt = amount * self.stt_pct
        slippage = amount * self.slippage_pct
        return brokerage + gst + stt + slippage

    def round_trip_cost(self, buy_amount: float, sell_amount: float) -> float:
        """Total cost for a complete buy + sell cycle."""
        return self.buy_cost(buy_amount) + self.sell_cost(sell_amount)

    def effective_buy_price(self, price: float) -> float:
        """Price after adding slippage (buying higher)."""
        return price * (1 + self.slippage_pct)

    def effective_sell_price(self, price: float) -> float:
        """Price after subtracting slippage (selling lower)."""
        return price * (1 - self.slippage_pct)


# =============================================================================
# Portfolio
# =============================================================================

class Portfolio:
    """
    Full portfolio state with 80/20 equity/debt allocation.

    Manages:
    - Active positions with sector tracking
    - Debt fund reserve with daily accrual
    - Position sizing and sector limits
    - Topup execution from debt reserve
    - Exit processing with cost calculation
    """

    def __init__(self, config: MQBacktestConfig):
        self.config = config
        self.cost_model = TransactionCostModel(config)

        # Capital pools
        self.equity_cash: float = 0.0  # Undeployed equity cash (starts at 0 after initial buy)
        self.debt_fund_balance: float = config.debt_reserve_capital
        self.debt_fund_initial: float = config.debt_reserve_capital

        # Positions
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.topup_log: List[TopupRecord] = []

        # Daily tracking
        self.daily_equity: Dict[str, float] = {}  # date_str -> total portfolio value
        self.daily_debt_fund: Dict[str, float] = {}  # date_str -> debt fund balance

        # Debt fund last accrual date
        self._last_accrual_date: Optional[datetime] = None

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def equity_value(self) -> float:
        """Total value of all stock positions."""
        return sum(p.current_value for p in self.positions.values())

    @property
    def total_value(self) -> float:
        """Total portfolio value (equity + cash + debt fund)."""
        return self.equity_value + self.equity_cash + self.debt_fund_balance

    @property
    def position_count(self) -> int:
        return len(self.positions)

    @property
    def sector_weights(self) -> Dict[str, float]:
        """Current sector weights as fraction of equity value."""
        total = self.equity_value
        if total <= 0:
            return {}
        weights = {}
        for p in self.positions.values():
            weights[p.sector] = weights.get(p.sector, 0) + p.current_value
        return {s: v / total for s, v in weights.items()}

    @property
    def sector_counts(self) -> Dict[str, int]:
        """Number of stocks per sector."""
        counts = {}
        for p in self.positions.values():
            counts[p.sector] = counts.get(p.sector, 0) + 1
        return counts

    @property
    def debt_fund_pct_remaining(self) -> float:
        """What % of original debt reserve is still available."""
        if self.debt_fund_initial <= 0:
            return 0
        return self.debt_fund_balance / self.debt_fund_initial

    # -------------------------------------------------------------------------
    # Debt Fund Accrual
    # -------------------------------------------------------------------------

    def accrue_debt_fund(self, current_date: datetime):
        """
        Accrue daily interest on the debt fund balance.

        Called once per trading day by the backtest engine.
        """
        if self._last_accrual_date is None:
            self._last_accrual_date = current_date
            return

        days = (current_date - self._last_accrual_date).days
        if days <= 0:
            return

        # Daily rate from annual
        daily_rate = (1 + self.config.debt_fund_annual_return) ** (1 / 365) - 1
        self.debt_fund_balance *= (1 + daily_rate) ** days
        self._last_accrual_date = current_date

    # -------------------------------------------------------------------------
    # Sector Limit Checks
    # -------------------------------------------------------------------------

    def can_add_to_sector(self, sector: str) -> bool:
        """Check if adding a stock to this sector would violate limits."""
        count = self.sector_counts.get(sector, 0)
        if count >= self.config.max_stocks_per_sector:
            return False

        # Weight check: use target equity capital as denominator during construction,
        # actual equity value once fully deployed. This prevents blocking during
        # initial portfolio build-up when only a few positions exist.
        denominator = max(self.equity_value, self.config.equity_capital)
        if denominator > 0:
            sector_value = sum(
                p.current_value for p in self.positions.values() if p.sector == sector
            ) + self.config.position_capital
            weight = sector_value / denominator
            if weight > self.config.max_sector_weight:
                return False

        return True

    def get_sector_violations(self) -> List[str]:
        """Return list of sector limit violation descriptions."""
        violations = []
        weights = self.sector_weights
        counts = self.sector_counts

        for sector, weight in weights.items():
            if weight > self.config.max_sector_weight:
                violations.append(
                    f"{sector}: {weight:.1%} weight (limit {self.config.max_sector_weight:.0%})"
                )
        for sector, count in counts.items():
            if count > self.config.max_stocks_per_sector:
                violations.append(
                    f"{sector}: {count} stocks (limit {self.config.max_stocks_per_sector})"
                )

        return violations

    # -------------------------------------------------------------------------
    # Entry
    # -------------------------------------------------------------------------

    def enter_position(
        self,
        symbol: str,
        sector: str,
        price: float,
        date: datetime,
        capital: float = None,
    ) -> Optional[Position]:
        """
        Open a new position.

        Args:
            symbol: Stock symbol
            sector: Stock sector
            price: Entry price
            date: Entry date
            capital: Capital to deploy (default: equal-weight allocation)

        Returns:
            Position if successful, None if blocked
        """
        if symbol in self.positions:
            logger.warning(f"Already holding {symbol}, skipping entry")
            return None

        if not self.can_add_to_sector(sector):
            logger.info(f"Sector limit reached for {sector}, skipping {symbol}")
            return None

        capital = capital or self.config.position_capital

        # Apply transaction costs
        buy_cost = self.cost_model.buy_cost(capital)
        effective_price = self.cost_model.effective_buy_price(price)
        investable = capital - buy_cost
        shares = int(investable / effective_price)

        if shares <= 0:
            return None

        actual_investment = shares * effective_price
        total_cost = actual_investment + buy_cost

        position = Position(
            symbol=symbol,
            sector=sector,
            entry_date=date,
            entry_price=price,
            shares=shares,
            initial_investment=actual_investment,
            current_price=price,
            rolling_ath=price,
        )

        self.positions[symbol] = position
        self.equity_cash -= total_cost  # Deduct full capital deployed + transaction costs

        logger.debug(
            f"ENTRY: {symbol} @ {price:.2f}, {shares} shares, "
            f"invested={actual_investment:.0f}, cost={buy_cost:.0f}"
        )
        return position

    # -------------------------------------------------------------------------
    # Exit
    # -------------------------------------------------------------------------

    def exit_position(
        self,
        symbol: str,
        price: float,
        date: datetime,
        reason: ExitReason,
    ) -> Optional[Trade]:
        """
        Close a position and record the trade.

        Returns sale proceeds to equity_cash after costs.

        Returns:
            Trade record if successful, None if position not found
        """
        position = self.positions.get(symbol)
        if position is None:
            return None

        effective_price = self.cost_model.effective_sell_price(price)
        gross_proceeds = position.total_shares * effective_price
        sell_cost = self.cost_model.sell_cost(gross_proceeds)
        net_proceeds = gross_proceeds - sell_cost

        gross_pnl = gross_proceeds - position.total_invested
        buy_cost_estimate = self.cost_model.buy_cost(position.total_invested)
        total_costs = buy_cost_estimate + sell_cost
        net_pnl = gross_pnl - total_costs
        return_pct = net_pnl / position.total_invested if position.total_invested > 0 else 0

        trade = Trade(
            symbol=symbol,
            sector=position.sector,
            entry_date=position.entry_date,
            entry_price=position.entry_price,
            shares_entered=position.shares,
            initial_investment=position.initial_investment,
            exit_date=date,
            exit_price=price,
            exit_reason=reason,
            exit_value=net_proceeds,
            topup_count=position.topup_count,
            total_invested=position.total_invested,
            total_shares_at_exit=position.total_shares,
            gross_pnl=round(gross_pnl, 2),
            transaction_costs=round(total_costs, 2),
            net_pnl=round(net_pnl, 2),
            return_pct=round(return_pct, 4),
        )

        self.trades.append(trade)
        self.equity_cash += net_proceeds
        del self.positions[symbol]

        logger.debug(
            f"EXIT [{reason.value}]: {symbol} @ {price:.2f}, "
            f"PnL={net_pnl:+,.0f} ({return_pct:+.1%})"
        )
        return trade

    # -------------------------------------------------------------------------
    # Topup
    # -------------------------------------------------------------------------

    def calculate_topup(
        self,
        symbol: str,
        price: float,
        date: datetime,
    ) -> Dict:
        """
        Calculate if a topup is possible and how much.

        Returns:
            Dict with 'can_topup', 'amount', 'shares', 'blocked_reason'
        """
        position = self.positions.get(symbol)
        if position is None:
            return {'can_topup': False, 'amount': 0, 'shares': 0,
                    'blocked_reason': 'No position found'}

        # Cooldown check
        if position.last_topup_date:
            days_since = (date - position.last_topup_date).days
            if days_since < self.config.topup_cooldown_days:
                return {'can_topup': False, 'amount': 0, 'shares': 0,
                        'blocked_reason': f'Cooldown ({days_since}/{self.config.topup_cooldown_days} days)'}

        # Calculate topup amount (20% of original entry)
        topup_amount = position.initial_investment * self.config.topup_pct_of_initial

        # Check debt fund has enough
        if self.debt_fund_balance <= 0:
            return {'can_topup': False, 'amount': 0, 'shares': 0,
                    'blocked_reason': 'Debt fund exhausted'}

        if self.debt_fund_balance < topup_amount:
            topup_amount = self.debt_fund_balance  # Deploy whatever is left

        # Position size cap check
        new_value = position.current_value + topup_amount
        max_allowed = self.total_value * self.config.max_position_size

        if new_value > max_allowed:
            topup_amount = max(0, max_allowed - position.current_value)
            if topup_amount <= 0:
                return {'can_topup': False, 'amount': 0, 'shares': 0,
                        'blocked_reason': 'Position size cap (10%) reached'}

        # Calculate shares
        effective_price = self.cost_model.effective_buy_price(price)
        buy_cost = self.cost_model.buy_cost(topup_amount)
        investable = topup_amount - buy_cost
        shares = int(investable / effective_price)

        if shares <= 0:
            return {'can_topup': False, 'amount': 0, 'shares': 0,
                    'blocked_reason': 'Topup amount too small'}

        return {
            'can_topup': True,
            'amount': round(topup_amount, 2),
            'shares': shares,
            'buy_cost': round(buy_cost, 2),
            'blocked_reason': None,
        }

    def execute_topup(
        self,
        symbol: str,
        price: float,
        date: datetime,
    ) -> Optional[TopupRecord]:
        """
        Execute a topup: deduct from debt fund, add shares to position.

        Returns:
            TopupRecord if successful, None if blocked
        """
        calc = self.calculate_topup(symbol, price, date)
        if not calc['can_topup']:
            return None

        position = self.positions[symbol]
        amount = calc['amount']
        shares = calc['shares']
        buy_cost = calc['buy_cost']

        debt_before = self.debt_fund_balance
        self.debt_fund_balance -= amount
        actual_investment = amount - buy_cost

        position.add_topup(shares, price, actual_investment, date)

        record = TopupRecord(
            symbol=symbol,
            date=date,
            price=price,
            shares=shares,
            amount=round(amount, 2),
            debt_fund_before=round(debt_before, 2),
            debt_fund_after=round(self.debt_fund_balance, 2),
            position_value_after=round(position.current_value, 2),
            topup_number=position.topup_count,
        )
        self.topup_log.append(record)

        logger.debug(
            f"TOPUP #{position.topup_count}: {symbol} @ {price:.2f}, "
            f"+{shares} shares, amount={amount:,.0f}, "
            f"debt_fund={self.debt_fund_balance:,.0f}"
        )
        return record

    # -------------------------------------------------------------------------
    # Exit Checks
    # -------------------------------------------------------------------------

    def check_hard_stop(self, symbol: str) -> bool:
        """Check if position has breached hard stop loss (30% from avg entry)."""
        position = self.positions.get(symbol)
        if position is None:
            return False
        return position.loss_from_entry >= self.config.hard_stop_loss

    def check_ath_drawdown_exit(self, symbol: str) -> bool:
        """Check if position has >20% drawdown from rolling ATH (rebalance exit)."""
        position = self.positions.get(symbol)
        if position is None:
            return False
        return position.drawdown_from_ath >= self.config.rebalance_ath_drawdown

    def update_fundamental_streak(self, symbol: str, quarter_declined: bool):
        """
        Update the quarterly decline streak for fundamental exit monitoring.

        Args:
            symbol: Stock symbol
            quarter_declined: True if this quarter's revenue declined >10% YoY
        """
        position = self.positions.get(symbol)
        if position is None:
            return

        if quarter_declined:
            position.quarterly_decline_streak += 1
        else:
            position.quarterly_decline_streak = 0  # Reset on recovery

    def check_fundamental_exit(self, symbol: str) -> bool:
        """Check if 3 consecutive quarters declined (fundamental exit trigger)."""
        position = self.positions.get(symbol)
        if position is None:
            return False
        return position.quarterly_decline_streak >= self.config.quarterly_decline_count

    # -------------------------------------------------------------------------
    # Daily Update
    # -------------------------------------------------------------------------

    def update_prices(self, prices: Dict[str, float], current_date: datetime):
        """
        Update all position prices and record daily portfolio value.

        Args:
            prices: Dict of symbol -> close price
            current_date: Current date
        """
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update_price(prices[symbol])

        # Accrue debt fund interest
        self.accrue_debt_fund(current_date)

        # Record daily values
        date_str = current_date.strftime('%Y-%m-%d')
        self.daily_equity[date_str] = self.total_value
        self.daily_debt_fund[date_str] = self.debt_fund_balance

    # -------------------------------------------------------------------------
    # Snapshot
    # -------------------------------------------------------------------------

    def snapshot(self) -> Dict:
        """Current portfolio state as a dict."""
        return {
            'total_value': round(self.total_value, 2),
            'equity_value': round(self.equity_value, 2),
            'equity_cash': round(self.equity_cash, 2),
            'debt_fund': round(self.debt_fund_balance, 2),
            'debt_fund_pct': round(self.debt_fund_pct_remaining * 100, 1),
            'positions': self.position_count,
            'trades_completed': len(self.trades),
            'topups_executed': len(self.topup_log),
            'sector_weights': {s: round(w * 100, 1) for s, w in self.sector_weights.items()},
            'sector_counts': self.sector_counts,
        }

    def get_position_summary(self) -> List[Dict]:
        """Summary of all active positions."""
        return sorted([
            {
                'symbol': p.symbol,
                'sector': p.sector,
                'entry_date': p.entry_date.strftime('%Y-%m-%d'),
                'entry_price': p.entry_price,
                'current_price': p.current_price,
                'shares': p.total_shares,
                'value': round(p.current_value, 2),
                'pnl': round(p.unrealized_pnl, 2),
                'return_pct': round(p.unrealized_return_pct * 100, 1),
                'drawdown_from_ath': round(p.drawdown_from_ath * 100, 1),
                'topups': p.topup_count,
            }
            for p in self.positions.values()
        ], key=lambda x: x['return_pct'], reverse=True)
