# Covered Calls Backtest & Research

A Python-based backtesting and research tool for Covered Call options strategies.

<!-- LAUNCHPAD:START -->
```json
{
  "stage": "building",
  "progress": 20,
  "complexity": "F",
  "lastUpdated": "2026-01-17",
  "targetDate": null,
  "nextAction": "Implement backtest engine core",
  "blocker": null,
  "demoUrl": null,
  "techStack": ["Python", "Pandas", "NumPy", "Jupyter", "Matplotlib"],
  "shipped": false,
  "linkedinPosted": false
}
```
<!-- LAUNCHPAD:END -->

## Overview

This project enables:
- Historical backtesting of covered call strategies
- Strike selection optimization (delta-based, % OTM)
- Performance metrics analysis (Sharpe, drawdown, returns)
- Visualization of equity curves and results

## Project Structure

```
Covered_Calls/
├── docs/                    # Documentation
│   ├── APP_PRD.md           # Product requirements
│   ├── DEV-CLOCK.md         # Development time tracking
│   ├── PROJECT-STATUS.md    # 9-step workflow status
│   └── GLOSSARY.md          # Domain terminology
├── data/                    # Historical data files
├── notebooks/               # Jupyter research notebooks
├── src/                     # Source code
│   ├── data/                # Data loaders
│   ├── strategies/          # Strategy implementations
│   ├── backtest/            # Backtest engine
│   ├── metrics/             # Performance calculations
│   └── visualization/       # Charting utilities
├── tests/                   # Unit tests
├── inits_n_info/            # Project setup info
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Start Jupyter for research
jupyter notebook

# Run tests
pytest tests/
```

## Development Workflow

This project follows the 9-step development workflow:

1. DEV-CLOCK - Time tracking
2. PRD & Design - Requirements definition
3. Test Cases - Test planning
4. Build - Implementation
5. Manual Testing - Validation
6. Debug & Feedback - Bug fixes
7. Code Walkthrough - Review
8. Ship - Deployment/Release
9. Time Retrospective - Analysis

See [docs/PROJECT-STATUS.md](docs/PROJECT-STATUS.md) for current status.

## License

Private - All rights reserved
