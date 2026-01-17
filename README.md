<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/NumPy-Math-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/Jupyter-Research-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" />
  <img src="https://img.shields.io/badge/License-Private-red?style=for-the-badge" alt="License" />
</p>

<h1 align="center">Covered Calls</h1>

<h3 align="center">
  Backtest & Research. <em>Data-driven options strategies.</em>
</h3>

<p align="center">
  A Python-based backtesting and research tool for Covered Call options strategies.<br />
  Strike optimization, performance metrics, and visualization.
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#tech-stack">Tech Stack</a> •
  <a href="#usage">Usage</a>
</p>

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

---

## Features

- **Historical Backtesting** — Test covered call strategies against historical data
- **Strike Selection** — Delta-based and % OTM optimization algorithms
- **Performance Metrics** — Sharpe ratio, max drawdown, CAGR, win rate
- **Visualization** — Equity curves, return distributions, comparative charts
- **Research Notebooks** — Jupyter-based exploratory analysis

---

## Quick Start

### Prerequisites

- Python 3.11+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/castroarun/Covered_Calls.git
cd Covered_Calls

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| [Python 3.11](https://python.org/) | Core language |
| [Pandas](https://pandas.pydata.org/) | Data manipulation |
| [NumPy](https://numpy.org/) | Numerical computing |
| [Jupyter](https://jupyter.org/) | Research notebooks |
| [Matplotlib](https://matplotlib.org/) | Visualization |

---

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

---

## Usage

```bash
# Start Jupyter for research
jupyter notebook

# Run tests
pytest tests/
```

---

## Roadmap

- [x] Project structure setup
- [x] Product requirements document
- [ ] Data loader implementation
- [ ] Backtest engine core
- [ ] Strike selection algorithms
- [ ] Performance metrics
- [ ] Visualization module
- [ ] Research notebooks

---

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

---

## License

Private - All rights reserved

---

<p align="center">
  <sub>Built by <a href="https://github.com/castroarun">Arun Castro</a></sub>
</p>