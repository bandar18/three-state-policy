# Multi-Asset Trading Classification System

A supervised learning system for predicting SPY returns using multi-asset features. This system implements Bull/Bear/Flat classification with comprehensive risk management and performance evaluation.

## Overview

The system predicts the probability that SPY's aggregated return over the next 15 minutes will be:
- **Bull class**: ≥ +0.02%
- **Bear class**: ≤ -0.02%
- **Flat class**: otherwise

It uses a rolling 75-minute feature window and makes predictions every 5 minutes using multi-asset data (SPY, VIX, DXY).

## Features

- **Data Sources**: SPY, VIX, DXY from Yahoo Finance
- **Feature Engineering**: 
  - Past return ladders (1, 5, 15 periods)
  - Rolling volatility and realized skewness
  - Volume z-scores
  - Technical indicators (RSI, MACD, ATR)
  - Cross-asset spreads
- **Model**: Gradient Boosting Classifier with time series cross-validation
- **Risk Management**: Stop-loss, drawdown limits, volatility filters
- **Performance Evaluation**: Sharpe ratio, Sortino ratio, confusion matrix, P&L analysis

## Installation

1. Install Python 3.11 or higher
2. Install TA-Lib (required for technical indicators):
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install libta-lib-dev
   
   # On macOS with Homebrew
   brew install ta-lib
   
   # On Windows, download from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Simply run the main script:

```bash
python trading_classifier.py
```

The system will:
1. Download the last 60 days of 1-minute data for SPY, VIX, and DXY
2. Build comprehensive features and labels
3. Train a gradient boosting classifier with grid search
4. Perform walk-forward backtesting
5. Calculate performance metrics
6. Generate plots and save them to `./figs/`

## Output

The system generates:

### Performance Metrics
- Total trades and win rate
- Total P&L and average P&L per trade
- Sharpe and Sortino ratios
- Maximum drawdown
- Long/short hit rates
- Balanced accuracy

### Plots
- Cumulative P&L curve (`./figs/performance_curves.png`)
- Drawdown analysis
- P&L distribution
- Position analysis
- Confusion matrix (`./figs/confusion_matrix.png`)

## Key Parameters

- **Feature Window**: 75 minutes (15 periods of 5-minute bars)
- **Prediction Horizon**: 15 minutes (3 periods)
- **Return Threshold**: 0.02% (0.05 × average bid-ask spread)
- **VIX Filter**: Skip trades when VIX > 25
- **Stop Loss**: 1 × ATR(20) from entry
- **Max Drawdown**: 5% limit

## Architecture

The system is organized into the following key methods:

1. **`load_data()`**: Downloads and aligns multi-asset data
2. **`build_features()`**: Creates comprehensive feature matrix
3. **`make_labels()`**: Generates Bull/Bear/Flat classifications
4. **`train_model()`**: Trains gradient boosting classifier
5. **`walk_forward_backtest()`**: Performs expanding window validation
6. **`risk_metrics()`**: Calculates performance statistics
7. **`plot_curves()`**: Generates visualization plots
8. **`main()`**: Orchestrates the entire pipeline

## Example Output

```
==================================================
Multi-Asset Trading Classification System
==================================================
Loading market data...
Loaded 8640 1-minute bars
Date range: 2024-01-01 09:30:00 to 2024-03-01 16:00:00
Building features...
Built 1200 feature rows with 228 features
Creating labels...
Label distribution:
Bull: 380
Bear: 375
Flat: 445
Training model...
Best parameters: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 300, 'subsample': 0.8}
Best CV score: 0.3456
Running walk-forward backtest...

==================================================
PERFORMANCE REPORT
==================================================
Total Trades: 156
Winning Trades: 78
Losing Trades: 78
Win Rate: 50.00%
Total P&L: 0.0234
Average P&L per Trade: 0.0001
Max Drawdown: -0.0045
Sharpe Ratio: 1.23
Sortino Ratio: 1.67
Long Hit Rate: 52.00%
Short Hit Rate: 48.00%
Long Trades: 82
Short Trades: 74
Average Balanced Accuracy: 0.3456
```

## Risk Considerations

- This is a demonstration system for educational purposes
- Real trading involves significant risk of loss
- Past performance does not guarantee future results
- Consider transaction costs, slippage, and market impact
- Test thoroughly before any live implementation

## Dependencies

- Python 3.11+
- yfinance: Market data downloading
- pandas: Data manipulation
- numpy: Numerical computing
- scikit-learn: Machine learning
- TA-Lib: Technical indicators
- matplotlib: Plotting
- seaborn: Statistical visualization

## License

This project is for educational purposes only. Use at your own risk.
