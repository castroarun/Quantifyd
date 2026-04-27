<div align="center">

# The Morning Note

**Pre-market intelligence — written by AI, dispatched at 08:00 IST.**
Three stages, ~3-minute loop, dual-rail delivery (Gmail + WhatsApp).

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![APScheduler](https://img.shields.io/badge/APScheduler-cron-58a6ff)](https://apscheduler.readthedocs.io)
[![yfinance](https://img.shields.io/badge/yfinance-Yahoo-purple)](https://github.com/ranaroussi/yfinance)
[![Claude Code](https://img.shields.io/badge/Claude_Code-cloud_routine-A78BFA)](https://claude.com/code)
[![Gmail SMTP](https://img.shields.io/badge/Gmail-SMTP-EA4335?logo=gmail&logoColor=white)](https://support.google.com/mail/answer/7126229)
[![Twilio](https://img.shields.io/badge/Twilio-WhatsApp-F22F46?logo=twilio&logoColor=white)](https://www.twilio.com/whatsapp)
[![Status: Live](https://img.shields.io/badge/Status-Live-22c55e)](https://castroarun.github.io/Quantifyd/premarket_brief_pipeline.html)

[**Pipeline visualization →**](https://castroarun.github.io/Quantifyd/premarket_brief_pipeline.html) ·
[Source: `services/premarket_brief.py`](../services/premarket_brief.py)

</div>

<!-- LAUNCHPAD:START -->
```json
{
  "stage": "live",
  "progress": 100,
  "complexity": "F",
  "lastUpdated": "2026-04-27",
  "targetDate": "2026-04-27",
  "nextAction": "Phase 2c: add earnings calendar + FII/DII flows",
  "blocker": null,
  "demoUrl": "https://castroarun.github.io/Quantifyd/premarket_brief_pipeline.html",
  "techStack": ["Python", "APScheduler", "yfinance", "Claude Code", "Gmail SMTP", "Twilio"],
  "shipped": true,
  "linkedinPosted": false
}
```
<!-- LAUNCHPAD:END -->

## What it does

Every weekday at 08:00 IST, you receive a clean pre-market intelligence brief in your inbox **and** WhatsApp — written by a cloud LLM, sourced from live exchange feeds, delivered before market open. The whole loop closes in three minutes.

## The three stages

| | When | Where | What |
|---|---|---|---|
| **Stage 1** | 08:00 IST | VPS (Contabo Linux) | Builds the JSON: 14 yfinance tickers (GIFT Nifty, S&P, USDINR, India VIX, Brent, Gold, etc.), F&O ban CSV, holdings calendar from `holdings_events.db`, top 25 RSS items deduped from Moneycontrol/Mint/Reuters India |
| **Stage 2** | 08:02 IST | Claude Code cloud routine | Curls the raw JSON, picks 5 most market-relevant headlines, tags each `POS`/`NEG`/`NEU`, writes the narrative One-Liner with key numbers highlighted, posts back |
| **Stage 3** | 08:03 IST | VPS | Renders the Mock C v5 HTML email, dispatches to **Gmail** (HTML) and **WhatsApp** (Twilio plain-text summary) |
| **Fallback** | 08:08 IST | VPS | If Stage 2's POST never arrived, the un-synthesized version goes out anyway — the operator never gets nothing |

## Why this architecture

**Each layer does one thing.** The VPS is good at deterministic work — fetching, persisting, transport. The cloud routine is good at the part that benefits from language understanding — sentiment, narrative, judgment. Mixing layers (e.g., letting the cloud send email directly) introduces failure modes that aren't worth it.

**The cloud routine is stateless.** No SSH keys, no SMTP credentials, no Kite tokens — nothing valuable to leak. The worst-case payload is a public-data forecast.

**Self-healing by design.** Cloud sandboxes can fail silently — model latency, network issues, sandbox spawning failures. The 08:08 IST fallback runs on the VPS regardless, so the operator always gets a brief. Idempotent: if Stage 3 already shipped, the fallback exits cleanly without sending a duplicate.

## Data sources (all free)

```
yfinance · 14 tickers
  ├─ GIFT_NIFTY, SP500, NASDAQ, DOW, US_VIX
  ├─ USDINR, INDIA_VIX, BRENT, GOLD
  ├─ UST10Y, DXY
  └─ NIKKEI, HSI, KOSPI

NSE archive
  └─ F&O securities-in-ban CSV (daily)

BSE corporate actions API
  └─ Forthcoming results (planned, Phase 2c)

RSS feeds
  ├─ Moneycontrol Markets
  ├─ Mint Markets
  └─ Reuters India Markets

holdings_events.db (local)
  └─ Today's results / dividends / splits / bonuses
     for every holding, refreshed at 07:00 IST
```

## API contract

The two halves communicate through a single JSON shape:

```bash
# Stage 2 reads:
GET  http://<vps>:5000/api/premarket/brief/raw

# Stage 2 writes back:
POST http://<vps>:5000/api/premarket/brief/synthesized
Content-Type: application/json

{
  "narrative_summary": "Friday's US strength flows into Asia. NIFTY indicated <b style='color:#16a34a'>+95 pts</b> on <b>+0.42%</b> GIFT premium. Only volatility flag: <b style='color:#1e40af'>Fed Powell · 18:00 IST</b>.",
  "headlines_synthesized": [
    {"tag": "POS", "text": "Fed minutes signal patient rate-cut path...", "source": "Bloomberg"},
    {"tag": "NEG", "text": "Saudi cuts 500k bpd...", "source": "Reuters"},
    ...
  ]
}
```

## Project structure

```
Quantifyd/
├── services/
│   └── premarket_brief.py          # Main module (data + render + dispatch)
├── scripts/
│   └── _gen_premarket_workflow_image.py  # Generates the diagram PNG
├── premarket_brief_pipeline.html   # GH Pages showcase
├── docs/
│   └── PREMARKET_BRIEF.md          # This file
└── backtest_data/
    └── premarket_brief_latest.json  # Last persisted brief (served via /api)
```

## Tech stack

| Component | Tech |
|---|---|
| Scheduler | **APScheduler** (in-process, inside gunicorn) |
| Market data | **yfinance** |
| RSS parser | **feedparser** |
| Holdings | local SQLite (`holdings_events.db`) |
| Cloud agent | **Claude Code** routine, `claude-sonnet-4-6` |
| Email | **Gmail SMTP** (TLS 587) |
| WhatsApp | **Twilio** REST API (`twilio` Python SDK) |
| Hosting | Contabo VPS (Ubuntu 24.04, gunicorn under systemd) |
| Visualization | inline SVG → cairosvg → PNG (1492×940) |

## Operational verification

The pre-market validator (separate `services/system_validator.py` job) runs at 08:50 IST daily and confirms `premarket_brief_build` and `premarket_brief_fallback` are both registered in the scheduler. The DB integrity watchdog checks `holdings_events.db` every 5 min during market hours. If anything breaks, the operator gets an alert before market open.

## Roadmap

- [x] **Phase 1** — KPI tiles, Market Context, Your Book Today, rule-based bias verdict
- [x] **Phase 2a** — RSS news fetcher, 2-stage scheduler, synthesis endpoint
- [x] **Phase 2b** — Claude Code cloud routine for sentiment + narrative
- [x] **Phase 2c** — WhatsApp delivery via Twilio
- [ ] **Phase 3** — Earnings calendar (BSE forthcoming results)
- [ ] **Phase 3** — FII/DII provisional flow widget
- [ ] **Phase 4** — Web preview at `/app/premarket` (React) for archive viewing

## License

Private — part of the Quantifyd platform.

---

<div align="center">
<sub>Designed and shipped 2026-04-27 · A small piece of a larger systematic-trading platform.</sub><br>
<sub><a href="https://castronix.dev">castronix.dev</a> · <a href="https://github.com/castroarun/Quantifyd">github.com/castroarun/Quantifyd</a></sub>
</div>
