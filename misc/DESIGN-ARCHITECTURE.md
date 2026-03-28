# Strategy Command Center — Design Architecture & Build Spec

## Overview

A standalone web application for monitoring and managing multiple automated trading strategies. The user logs in and gets a unified cockpit view of all running systems, positions, P&L, capital deployment, and trade history — with alerts surfaced inline wherever they are, not hidden behind separate tabs.

**Core philosophy: Information comes to you, you don't go hunting for it.**

---

## Tab Structure (Only 4 Tabs)

### Tab 1: 📈 Dashboard (Home)
The "5-second rule" view. Everything critical at a glance.

**Sections:**
1. **Metric Cards Row** — Today's P&L, Total P&L, Deployed Capital, Free Cash, Active Strategies count, Open Positions count
2. **Capital Deployment Bar** — Overall utilization percentage with color-coded bar (green <60%, amber 60-80%, red >80%). Per-strategy mini utilization bars below it.
3. **Charts Row** — Weekly P&L bar chart, Monthly P&L (6M) bar chart, Strategy quick controls (play/pause toggles with live P&L per strategy)
4. **Recent Journal Notes** — Latest 3 journal entries from trades, surfaced here automatically
5. **Alert Banner** (conditional) — Red banner at top when any flags exist, appears on ALL tabs

**Key behaviors:**
- Strategy cards show error state with red background + ⚠️ when in error
- Play/Pause toggle on each strategy — single click
- Capital bar turns red when utilization >80%
- Privacy mode blurs all financial data

### Tab 2: 📊 Positions & Trades (Merged)
Two sub-views toggled by a segmented control: **Live** and **History**

#### Live Sub-view
- Full positions table with columns: Flag indicator, Strategy, Symbol, Side, Qty, Entry, Current, P&L (₹), P&L (%), SL, TP, Close button
- **Row-level flags**: Positions with issues get ⚠️ icon, red-tinted row background, and inline flag badges directly under the symbol (e.g., "Approaching SL", "Loss > ₹5K")
- **Close button** with confirmation step (click → "Exit / No" appears)
- **Strategy P&L summary strip** below the table — compact cards showing each strategy's daily P&L, risk badge, and Sharpe ratio

#### History Sub-view
- Filterable: All / Wins / Losses toggle
- Trade cards showing: Symbol, Side, Strategy tag, Date, Holding period, Entry/Exit, P&L (₹ and %), Trade notes
- **Inline journal** on every trade card:
  - Empty trades show "+ Add journal note" link
  - Click opens a textarea right inside the card
  - Saved journals display as amber boxes, clickable to edit
  - No separate journal page — notes live with the trades they belong to

### Tab 3: 📐 Strategy Blueprints
Standardized rule framework for every trading system.

**Strategy selector** — horizontal pills showing each strategy with live status badge

**Each blueprint contains:**

**Header section:**
- Strategy name, version tag (e.g., v2.4), System Type badge (Intraday/Swing/Positional/Monthly)
- Description paragraph
- GitHub link button (opens repo in new tab)
- Metadata row: Holding period, Timeframe, Instruments, Capital allocated
- Last commit info (date + message)

**Two views toggled by buttons:**

#### 📋 Rules & Criteria View
Three-column layout:
1. **Entry Rules** (green-coded) — numbered list, ALL conditions must be true
2. **Exit Rules** (blue-coded) — numbered list, ANY condition triggers exit
3. **Stop Loss Rules** (red-coded) — numbered list, layered protection

Below the three columns:
- **Position Sizing** — highlighted in amber
- **Indicators Used** — monospaced pills
- **Filters & Conditions** — bordered pills
- **Tags** — purple pills (e.g., #momentum, #options, #beta)

**Privacy mode** blurs the entire rules section (but header/GitHub link remain visible)

#### 📊 Backtest Metrics View
- **Key ratios row**: Total Trades, Win Rate, Profit Factor, Expectancy (R), Sharpe, Sortino, Calmar, Avg Holding
- **Drawdown Analysis** (red highlight box) — the critical section:
  - Max Strategy Drawdown
  - Average Strategy Drawdown
  - Max Drawdown per Trade (MAE)
  - Average Drawdown per Trade
  - Max consecutive losses
  - Average loss %
- **P&L profile**: Avg Win %, Avg Loss %, Best Month, Worst Month
- **Monthly expectation**: Average monthly return over backtest period

### Tab 4: 🔍 Day Deep Dive
Historical day-level snapshots with arrow navigation.

**Navigation:**
- ← → arrow buttons on screen
- ← → keyboard arrow keys also work
- Shows "Snapshot X/N" indicator

**Each day snapshot:**
1. **Summary cards**: Day P&L, Trades count (W/L split), Win Rate, Best Trade, Worst Trade, Capital Used
2. **Strategy P&L breakdown** — compact pills showing each strategy's contribution
3. **Full trade list** — every trade that day with Symbol, Strategy, Side, Entry/Exit, P&L, Time window, Notes

---

## Cross-Cutting Features

### Alert System (No Separate Tab)
Alerts surface inline wherever the user is:

1. **Global banner** — Red bar below header, visible on ALL tabs when flags exist. Shows:
   - Strategy error states (e.g., "Gap Fill Scalper: Max DD breached")
   - Position-level flags (e.g., "INFY: Near SL")
   - System-level warnings (e.g., "Capital utilization at 78%")
   - "View →" link to jump to Positions tab
2. **Pulsing red dot** in header when alerts are active
3. **Row-level flags** in positions table — red tinted rows with inline badge text
4. **Strategy error highlighting** — red background on strategy cards in Dashboard

**Alert rules (configurable):**
| Rule | Threshold | Action |
|------|-----------|--------|
| Max DD per Strategy | -10% | Auto-pause + alert |
| Daily Loss (All) | -₹25K | Alert banner |
| Capital Utilization | >80% | Warning |
| Single Position Loss | -₹10K | Alert + suggest close |
| Strategy Idle | >6 hours | Info |
| Consecutive Losses | 5 in a row | Auto-pause |

### Privacy Mode
- Toggle in header: 🔓 Open ↔ 🔒 Private
- When active, blurs: All P&L figures, positions table, capital data, strategy rules, day deep dive numbers
- Remains visible: Strategy names, status badges, play/pause controls, navigation
- Use CSS `filter: blur(8px)` + `pointer-events: none` + `user-select: none`

### Kill All (Emergency)
- Red "✕ KILL ALL" button always visible in header
- Opens confirmation modal with count of positions and strategies affected
- On confirm: closes all positions, stops all strategies
- Cannot be undone

---

## Data Model

### Strategy
```typescript
interface Strategy {
  id: string;
  name: string;
  type: "Equity" | "F&O";
  status: "running" | "paused" | "error" | "stopped";
  capital: number;       // allocated capital
  deployed: number;      // currently in use
  pnlToday: number;
  pnlTotal: number;
  winRate: number;        // percentage
  trades: number;         // total count
  maxDD: number;          // max drawdown %
  sharpe: number;
  lastSignal: string;     // relative time
  risk: "low" | "medium" | "high";
}
```

### Position
```typescript
interface Position {
  id: string;
  strategy: string;       // strategy name
  symbol: string;
  type: "EQ" | "OPT";
  side: "LONG" | "SHORT";
  qty: number;
  entry: number;
  current: number;
  pnl: number;
  pnlPct: number;
  sl: number;
  tp: number | null;
  flags: string[];        // inline alert messages
}
```

### Blueprint
```typescript
interface Blueprint {
  id: string;
  name: string;
  version: string;
  github: string;         // full repo URL
  lastCommit: string;     // date
  commitMsg: string;
  systemType: "Intraday" | "Swing" | "Positional" | "Monthly";
  holdingPeriod: string;
  timeframe: string;      // e.g., "15min", "Daily", "Weekly"
  instruments: string;
  capital: string;
  description: string;
  entry: string[];        // ALL must be true
  exit: string[];         // ANY triggers
  stopLoss: string[];     // layered protection
  positionSizing: string;
  indicators: string[];
  filters: string[];
  tags: string[];
  backtest: BacktestMetrics;
}

interface BacktestMetrics {
  period: string;
  totalTrades: number;
  winRate: number;
  avgWin: number;         // percentage
  avgLoss: number;        // negative percentage
  profitFactor: number;
  expectancy: number;     // in R-multiple
  maxDD: number;          // max strategy drawdown %
  avgDD: number;          // avg strategy drawdown %
  maxConsecLoss: number;
  avgHolding: string;
  sharpe: number;
  sortino: number;
  calmar: number;
  bestMonth: number;
  worstMonth: number;
  avgMonthly: number;
  maxDDTrade: number;     // max drawdown per trade (MAE)
  avgDDTrade: number;     // avg drawdown per trade
}
```

### Trade (History)
```typescript
interface Trade {
  id: string;
  date: string;
  strategy: string;
  symbol: string;
  side: "LONG" | "SHORT";
  entry: number;
  exit: number;
  pnl: number;
  pnlPct: number;
  holding: string;
  notes: string;          // system-generated or quick note
  journal: string;        // user's personal reflection
}
```

### DaySnapshot (for Deep Dive)
```typescript
interface DaySnapshot {
  date: string;
  dow: string;            // day of week
  pnl: number;
  trades: number;
  wins: number;
  losses: number;
  bestTrade: number;
  worstTrade: number;
  capitalUsed: number;
  items: DayTrade[];
}

interface DayTrade {
  symbol: string;
  strategy: string;
  side: "LONG" | "SHORT";
  pnl: number;
  pnlPct: number;
  entry: number;
  exit: number;
  timeWindow: string;     // e.g., "09:22–14:45"
  notes: string;
}
```

---

## Design System

### Theme: Light Mode Only
No dark mode. Professional, clean, airy.

### Colors
| Token | Value | Usage |
|-------|-------|-------|
| Background | `#f8fafc → #e4eaf1` gradient | Page background |
| Card | `#ffffff` | All card surfaces |
| Border | `#e2e8f0` | Card borders |
| Text Primary | `#0f172a` | Headings, values |
| Text Secondary | `#64748b` | Labels, meta |
| Text Muted | `#94a3b8` | Timestamps |
| Profit | `#059669` (emerald-700) | Positive P&L |
| Loss | `#dc2626` (red-600) | Negative P&L |
| Warning | `#f59e0b` (amber-500) | Warnings |
| Info | `#0ea5e9` (sky-500) | Informational |
| Kill Red | `#dc2626 → #b91c1c` gradient | Kill buttons |

### Typography
| Element | Font | Weight | Size |
|---------|------|--------|------|
| Headings | DM Sans | 700 | 18px (h2), 16px (h3) |
| Body | DM Sans | 400-600 | 14px |
| Labels | DM Sans | 600 | 12px, uppercase, tracking-wider |
| Financial values | JetBrains Mono | 600-700 | 14-20px |
| Badges | DM Sans | 700 | 12px |

### Components
| Component | Spec |
|-----------|------|
| Cards | `rounded-xl` or `rounded-2xl`, `border border-slate-200`, `shadow-sm`, `hover:shadow-md` |
| Buttons | `rounded-lg` or `rounded-xl`, font-semibold, 2px border |
| Status Badge | Pill shape, dot indicator, pulse animation for "running" |
| Risk Badge | Compact, uppercase, color-coded (emerald/amber/red) |
| System Type Badge | Rounded-full pill (violet=Intraday, sky=Swing, teal=Positional, rose=Monthly) |
| Tab Bar | Segmented control in `bg-slate-100`, active tab gets white bg + shadow |
| Table Rows | Minimal borders, hover highlight, red tint for flagged rows |
| Alert Banner | `bg-red-50 border-red-200 rounded-xl`, with 🚨 icon |
| Privacy Blur | `filter: blur(8px); user-select: none; pointer-events: none` |

### Currency Formatting
```typescript
// Indian format with Lakhs
const fmt = (n: number) => {
  const abs = Math.abs(n);
  if (abs >= 100000) return (n < 0 ? "-" : "") + "₹" + (abs / 100000).toFixed(2) + "L";
  return (n < 0 ? "-" : "") + "₹" + abs.toLocaleString("en-IN");
};
```

---

## Technical Recommendations

### Stack (Suggested)
- **Frontend**: React + Tailwind CSS (the mock is already in this)
- **State**: Zustand or React Context (lightweight)
- **Backend**: Node.js/Express or Python FastAPI
- **Database**: PostgreSQL (trades, positions, journal) + Redis (live state, caching)
- **Auth**: Simple JWT-based login (single user)
- **Broker API**: Zerodha Kite Connect for live positions & order management
- **Real-time**: WebSocket for live position updates
- **Deployment**: Self-hosted on a VPS or Raspberry Pi, or Vercel/Railway

### API Endpoints (Suggested)
```
GET    /api/dashboard           → metrics, capital, strategy states
GET    /api/positions           → live positions with flags
POST   /api/positions/:id/close → close a position
GET    /api/strategies          → all strategies with status
POST   /api/strategies/:id/toggle → pause/resume
POST   /api/strategies/kill-all → emergency kill
GET    /api/blueprints          → strategy rules & backtest data
GET    /api/trades?filter=...   → trade history
POST   /api/trades/:id/journal  → save/update journal note
GET    /api/days?offset=N       → day snapshots for deep dive
GET    /api/alerts              → computed alerts/flags
```

### File Structure (Suggested)
```
strategy-command-center/
├── src/
│   ├── app/
│   │   ├── layout.tsx
│   │   └── page.tsx
│   ├── components/
│   │   ├── Dashboard.tsx
│   │   ├── PositionsAndTrades.tsx
│   │   ├── Blueprints.tsx
│   │   ├── DayDeepDive.tsx
│   │   ├── AlertBanner.tsx          ← renders on all tabs
│   │   ├── StrategyCard.tsx
│   │   ├── PositionRow.tsx          ← with inline flags
│   │   ├── TradeCard.tsx            ← with inline journal
│   │   ├── BlueprintRules.tsx
│   │   ├── BlueprintBacktest.tsx
│   │   ├── CapitalBar.tsx
│   │   ├── KillAllModal.tsx
│   │   └── ui/                      ← StatusBadge, RiskBadge, MetricCard, MiniBar, etc.
│   ├── hooks/
│   │   ├── useStrategies.ts
│   │   ├── usePositions.ts
│   │   ├── useAlerts.ts             ← computed from positions + strategies
│   │   └── usePrivacy.ts
│   ├── lib/
│   │   ├── api.ts
│   │   ├── format.ts                ← fmt(), pct(), cls()
│   │   └── types.ts                 ← all TypeScript interfaces
│   └── store/
│       └── index.ts                 ← Zustand store
├── server/                           ← backend
│   ├── routes/
│   ├── services/
│   │   ├── zerodha.ts               ← Kite Connect integration
│   │   ├── alertEngine.ts           ← flag computation
│   │   └── strategyManager.ts
│   └── db/
│       ├── schema.sql
│       └── migrations/
├── public/
├── tailwind.config.js
├── package.json
└── README.md
```

---

## Strategies Included in Mock (6 Systems)

| # | Name | Type | Timeframe | Style |
|---|------|------|-----------|-------|
| 1 | Momentum Breakout | Equity/Swing | Daily | Trend-following breakouts with volume |
| 2 | Mean Reversion F&O | Options/Intraday | 15min | Keltner channel reversals |
| 3 | Keltner Squeeze | Options/Intraday | 15min | TTM Squeeze volatility breakout |
| 4 | Sector Rotation | Equity/Positional | Weekly | RS-based sector rotation |
| 5 | Gap Fill Scalper | Equity/Intraday | 5min | Counter-trend gap fills (beta) |
| 6 | Covered Call Writer | F&O/Monthly | Daily | Premium income via covered calls |

Each has full Entry/Exit/SL rules, position sizing, indicators, filters, tags, GitHub link, and backtested metrics in the mock.

---

## Implementation Priority

### Phase 1: Core Shell
- [ ] Auth (simple login)
- [ ] Dashboard with static/mock data
- [ ] 4-tab navigation
- [ ] Privacy mode toggle
- [ ] Kill All modal

### Phase 2: Live Data
- [ ] Zerodha Kite Connect integration
- [ ] Live positions with WebSocket updates
- [ ] Strategy status from your automation layer
- [ ] Computed alerts/flags engine

### Phase 3: History & Journal
- [ ] Trade history storage (PostgreSQL)
- [ ] Inline journal on trade cards
- [ ] Day deep dive with arrow navigation
- [ ] Trade filtering (all/wins/losses)

### Phase 4: Blueprints
- [ ] Blueprint CRUD (add/edit strategies)
- [ ] GitHub integration (pull last commit info)
- [ ] Backtest metrics import (from your backtesting output)

### Phase 5: Polish
- [ ] Mobile responsiveness
- [ ] Notifications (browser push / Telegram)
- [ ] Export reports (daily/weekly P&L PDF)
- [ ] Keyboard shortcuts
