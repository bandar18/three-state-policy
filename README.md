# three-state-policy 🐂🐻➖
Prototype repo for an  intraday bull / bear / flat classifier on $SPY, built with free `yfinance` data and a lightweight policy-head trading loop.
Problem Statement :-
Given the last N minutes of multi-asset features, estimate P(bull), P(bear), P(flat) for SPY’s 15-minute forward return and trade long/short/flat accordingly.
          | Dial |                                           | Setting |
| Feature window **N**        |    ----->       |     75 minutes (rolling)          |
| Prediction horizon **Δt**   |    ----->       |   15 minutes aggregated return    |
| Decision cadence            |    ----->       |          every 5 minutes          |
| Threshold **η**             |    ----->       |  0.05 × average bid-ask ≈ 0.02 %  |
| Back-test span              |    ----->       | last 60 trading days (1-min bars) |
