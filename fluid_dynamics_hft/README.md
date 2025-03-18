# Fluid Dynamics to HFT

A project applying fluid dynamics concepts to high-frequency trading strategies.

## Project Structure

```
fluid_dynamics_hft/
│
├── data/                      # Data storage and management
│   ├── raw/                   # Raw market data files (LOBSTER, etc.)
│   ├── processed/             # Preprocessed data
│   └── synthetic/             # Synthetic test data
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── fluid_market_model.py  # Market to fluid dynamics conversion
│   ├── pattern_identification.py  # Pattern recognition algorithms
│   ├── fluid_hft_strategy.py  # Trading strategy implementation
│   ├── fluid_market_system.py # End-to-end system integration
│   └── utils/                 # Utility functions
│       ├── __init__.py
│       ├── visualization.py   # Visualization tools
│       └── data_loaders.py    # Data loading utilities
│
├── notebooks/                 # Jupyter notebooks for exploration and demonstration
│   ├── 01_data_exploration.ipynb
│   ├── 02_fluid_model_visualization.ipynb
│   ├── 03_pattern_recognition.ipynb
│   └── 04_strategy_backtest.ipynb
│
├── tests/                     # Unit and integration tests
│   ├── __init__.py
│   ├── test_fluid_market_model.py
│   ├── test_pattern_identification.py
│   └── test_fluid_hft_strategy.py
│
├── configs/                   # Configuration files
│   ├── default_config.json    # Default system parameters
│   └── backtest_configs/      # Strategy-specific configs
│
├── outputs/                   # Analysis outputs
│   ├── figures/               # Saved visualizations
│   ├── reports/               # Generated reports
│   └── backtest_results/      # Strategy performance data
│
├── requirements.txt           # Dependencies
├── setup.py                   # Package setup
└── .gitignore                 # Git ignore file
```

## Installation

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Install the package in development mode: `pip install -e .`

## Usage

See the notebooks directory for examples and demonstrations.
