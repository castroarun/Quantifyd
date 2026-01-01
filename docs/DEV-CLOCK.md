# Covered_Calls - DEV-CLOCK

Time tracker for development phases. **Auto-updated from git commits.**

> Last updated: - | Total: **0 hours** | 0 commits

---

## Summary Statistics

| Phase | Estimated | Actual | Progress |
|-------|-----------|--------|----------|
| Design & Planning | 4h | 0h | ░░░░░░░░░░ 0% |
| Documentation | 2h | 0h | ░░░░░░░░░░ 0% |
| Research | 8h | 0h | ░░░░░░░░░░ 0% |
| Building | 16h | 0h | ░░░░░░░░░░ 0% |
| Debugging | 6h | 0h | ░░░░░░░░░░ 0% |
| Testing | 4h | 0h | ░░░░░░░░░░ 0% |
| **Total** | **40h** | **0h** | **0%** |

---

## Setup Instructions

### 1. Copy GitHub Action files to your project:

```
.github/
  workflows/
    dev-clock.yml
  scripts/
    update-dev-clock.js
```

### 2. Create docs/DEV-CLOCK.md (this file)

### 3. Push to GitHub - tracking starts automatically!

---

## Usage

### Normal commits (30-min buffer added automatically):
```bash
git commit -m "feat: add strategy backtest"
```

### Explicit start time (if you started at 8am):
```bash
git commit -m "feat: add covered call scanner [started:8am]"
```

### Session start marker:
```bash
git commit --allow-empty -m "wip: starting session [started:8am]"
```

---

## How It Works

- **Automatic tracking** from git commit timestamps
- **Session detection**: commits within 2hr gaps = same session
- **Phase detection**: parsed from commit message prefixes (feat:, fix:, docs:, etc.)
- **30-min buffer**: added before first commit of each session
- **Updates on every push** via GitHub Actions

---

## Phase Keywords

| Phase | Commit prefixes |
|-------|-----------------|
| Design & Planning | design, plan, rfc, spec, architecture |
| Documentation | docs, readme, doc: |
| Research | research, data, analysis, explore |
| Building | feat, feature, add, implement, create, build |
| Debugging | fix, bug, hotfix, patch, debug, resolve |
| Testing | test, spec, backtest, validate |

---

**Started:** -
**Status:** Pre-Development
