# Same-day slot swap — **NO EDGE**. Keep the one-day delay.

**research/157_same_day_slot_swap** · 08-Sep-2026 · Open Alpha (r/142 adopted 16-slot spec)

## The question

Arun: *"a slot freed by today's exit is available to the next day's scan — why the 1 day
delay? was it intentional? can u test the same day swap?"*

r/142's per-bar loop runs **entries, then exits at the close**, so a name that stops out on
bar *i* frees its slot only for bar *i+1*. Nothing in the study documents that as a
decision — the code is `# entries` … `# exits at close` with no comment — so it looked like
an artefact of block order.

## Verdict

**NO EDGE. The delay costs nothing measurable, and it stays.**

Both arms fill at the close, so block order is the only difference:

| | A · next-day | B · same-day | B − A |
|---|---|---|---|
| CAGR | 26.44% | 26.97% | **+0.53pp** |
| Max drawdown | −46.10% | −44.48% | +1.62pp |
| Calmar | 0.57 | 0.61 | +0.03 |
| Trades | 1,748 | 1,862 | +114 |
| Win rate | 39.6% | 38.8% | −0.76pp |
| Mean per trade | 2.23% | 2.18% | −0.05pp |

**B beat A in 4 of 7 years** — 2021 +21.0, 2022 +1.2, 2023 +1.9, 2020 +0.5, against 2024
−12.9, 2026 −0.3, 2025 −0.1. A half-point of CAGR that arrives on a 4-3 split, with the
per-trade average slightly *worse*, is not an edge. It is the same book taking 114 more
trades to end up in the same place.

## The part worth remembering: the first answer was +12.57pp, and it was false

The first run said B beat A by **+12.57pp of CAGR** (86.06% vs 73.50%), on a shallower
drawdown, in 6 of 7 years. It was **look-ahead**, and the cause was one line:

```python
fill = max(piv, float(O[i, c]))     # entries fill at the OPEN
...
if cl <= b * (1 - stop): ...        # exits happen at the CLOSE
```

Running exits first therefore sold at today's **close** and spent the proceeds at today's
**open**, hours earlier — the freed slot and the freed cash both arrived from the future.
The entire +12.57pp was foresight.

Once entries fill at the close, so that the money exists when it is spent, the advantage
collapses from +12.57pp to +0.53pp. **96% of the apparent edge was the bug.**

## Why the delay is not really a choice anyway

The live book enters on a **buy-stop at the pivot**, which fills at or near the open — the
`max(pivot, open)` mechanic, and the reason arm A's open-fill control compounds far faster
(73.50% CAGR) than either close-fill arm (~26%). A same-day swap is *incompatible* with
that mechanic: the exit is not known until the close, so there is no honest way to have
bought at that morning's pivot. To swap same-day the book would have to abandon the buy-stop
and buy at the close instead — and the table above says the swap wins nothing, while the
control gap says the close fill loses a great deal.

So the one-day delay is not a flaw to fix. It is what the entry mechanic already implies.

## What this settles for the build

The entry scanner runs **after the close** and places the replacement as a **buy-stop for
the next session** — not as part of the 15:18 exit pass. Exits and entries stay separate
jobs, which is also the simpler thing to operate.

## Controls

- **Fidelity control passed.** The forked `simulate` reproduced untouched
  `bluesky_replay.simulate` exactly — identical equity curve, 1,715 trades both. Without
  that, none of the above would mean anything.
- Selection is `rs` and deterministic: no seed ensemble is needed and no selection luck is
  being averaged away.
- 1,660 trading days, 12,025 signals, 2020-01-01 → 2026-09-05, 25 bps per side.

## Caveats stated plainly

- **Levels are inflated, deltas are not.** 2020→2026 is a smallcap bull run, the universe
  is chosen by symbols having ≥260 rows *today* (survivorship), and no tax is charged. Both
  arms carry the identical bias, which is why this is reported as a paired difference and
  the absolute CAGRs should not be quoted anywhere.
- One window only. The finding is that the difference is small, not that a precise number
  is right — a second window would not change "half a point on a 4-3 split".
