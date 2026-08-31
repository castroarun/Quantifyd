# V2 Iron-Fly — 1.5% Move-Stop Shadow Logger

STATUS: **COMMITTED — activates at the next 09:00 pre-open restart. Live stop UNCHANGED at 2.0%.**
Opened 2026-08-31 18:50 IST · market closed · no restart taken

---

## 1. The Ask

**What you asked:** "IF 1.5 shows better results than 2.0 SL, however the live paper
trading is doing 2.0 — please help understand this, or bring in the 1.5% live executor
also… reconstruct the trades against 1.5 side by side." Then, on the finding:
*"the cheap version is to log what the 1.5% stop would have done as a shadow field on
each live trade — yes pls do so."*

## 2. Why the live stop was NOT changed

The recorded-chain replay shows 1.5% ahead of 2.0% by ₹2.11L over 18 trades. That
does not survive a paired test on the same entries:

| | |
|---|---:|
| closed pairs | 17 |
| **outcomes identical** | **11** — the stop level never came into play |
| outcomes differing | 6 (1.5% better on 4, worse on 2) |
| mean difference | ₹14,984 per trade toward 1.5% |
| std dev | ₹50,832 |
| **paired t** | **1.22** — needs ~2.0 |
| largest single contributor | **₹1,54,862 = 61% of the entire gap** |
| **excluding that one trade** | mean ₹6,241, **t = 0.67** |

The mechanism cuts both ways in the same sample. On 26 May the tight stop exited at
−₹536 before a move that cost the loose stop −₹1,55,398. On 17 July the tight stop cut
a winner at +₹32,170 that the loose stop rode to **+₹1,04,651**. One big save, one big
miss, four small differences.

**Switching a live stop on t = 1.22 driven by a single trade is fitting to one
outcome.** The same lesson is already banked: research/96 removed the spot-move stop
from ATM2 entirely (`move_stop_pct: 0`) after finding it a DTE-dependent loss and an
expiry-gamma trap.

## 3. What was built instead

`services/v2_ironfly_api.py` gains a shadow logger following the two already in that
file (`compression_shadow_job`, the jade logger), both explicitly "changes no trading
behaviour":

- `SHADOW_STOP = 0.015`
- `_shadow_stop_check(pos, spot, pnl, ctime)` — records the **first** moment an open
  position travels 1.5% from entry: the time, the spot, the move %, and the position's
  mark at that instant, which is what the tighter stop would have realised.
- Called from `monitor_job()` **inside a try/except**, so a failure in the observation
  path can never reach the trading path below it.
- Table `v2_shadow_stop`, `pos_id` PRIMARY KEY + `INSERT OR IGNORE` so only the first
  breach is kept — the one a stop would have acted on.
- `GET /api/v2-ironfly/shadow-stops` — closed positions with actual exit beside the
  shadow outcome, plus the running total and mean effect.

**Nothing exits, nothing is sized, no order is placed.** Positions where the shadow
never triggers are identical to the live book by construction, which is the point:
only the differing trades accumulate.

## 4. Verification before commit

- `ast.parse` clean
- `_close(pos, "move2%")` still appears exactly once — the real stop is untouched
- `stop_pct=0.02` still present — the live level is unchanged
- 950 → 1,038 lines, all additive

## 5. Deployment

**No restart taken.** It was 18:50 IST — past the 15:40 gate and safe — but the
existing `0 9 * * 1-5 preopen_restart.sh` cron activates it before Tuesday's session,
so bouncing a live-adjacent process for a logger is unnecessary risk.

**Check on Tuesday:**

```bash
sudo journalctl -u quantifyd --since 09:00 | grep 'V2-shadow'
curl -s localhost:5000/api/v2-ironfly/shadow-stops | python3 -m json.tool | head -30
```

Expect silence until a position actually travels 1.5% — on the replay that is roughly
6 trades in 17, so a few weeks between entries.

## 6. What the evidence will settle, and when

The paired standard deviation is ₹50,832 against a mean difference of ₹14,984, so
reaching t = 2.0 needs about **46 differing trades**. At the observed rate — 6 of 17
positions ever breach 1.5% — that is roughly **130 positions**, well over a year.

That is the honest timeline, and it is the argument for logging rather than switching:
the question cannot be settled quickly, and the shadow costs nothing while it waits.
