# Apple Product Launch Backtesting

A Python-based toolkit to analyze how Apple’s stock (AAPL) reacts to product launches, identify repeatable short-term alpha signals, and visualize risk-adjusted performance.

## 🔍 Overview
- Performs an **event study** on 100+ Apple product launches (1980–2025)
- Calculates **CAPM-adjusted abnormal returns** and **Cumulative Abnormal Returns (CAR)**
- Runs a **grid search** of buy/sell offsets, ranks strategies by **Sharpe** & **Sortino** ratios
- Validates with **Monte Carlo** simulations, **VaR/CVaR**, max drawdown, and hit rate
- Generates equity-curve plots and Sharpe-ratio heatmaps by product family

## 🚀 Features
- **data_loader**: Fetches & cleans AAPL & S&P 500 data via yfinance
- **event_study**: Computes log returns, market relationship, abnormal returns, and CAR
- **strategy_backtest**: Backtests all buy/sell combinations, summarizes Sharpe/mean/std/hit rate
- **monte_carlo**: Bootstraps abnormal returns, simulates strategy distributions, computes CIs
- **risk_metrics**: Calculates VaR, CVaR, Sortino, drawdown, and other downside measures
- **viz**: Product-Based Sharpe heatmaps

## ▶️ Usage
3. **Explore the notebook for visualizations and stats**:
   ```bash
   jupyter lab Apple_Product_Launch_Backtest.ipynb
   ```
   > **Requirements:** Python 3.8+, pandas, numpy, scipy, statsmodels, matplotlib, seaborn, yfinance
3. **Key outputs**:
   - Top 5 launches by CAR per product family
   - Sharpe-ratio leaderboard & best buy/sell rules
   - Equity curve & Sharpe heatmap visualizations
   - Monte Carlo CIs, VaR/CVaR, drawdown summaries

## 📈 Results
- **AirPods** launches yielded the strongest alpha (Sharpe 0.98, Sortino 3.3) with a 75% hit rate
- **iPod** and **iPhone** also show positive event-driven strategies, while **Mac** moves are weaker
- Markets tend to **underreact** to novel categories, creating exploitable short-term inefficiencies

## 🔮 Future Improvements
- Extend analysis to rolling event windows and longer-term horizons (e.g., 30+ days post-launch)
- Add interactive dashboard (e.g., Tableau or Streamlit) for real-time strategy testing and visualization
- Automate data updates and incorporate new product launches dynamically
- Explore cross-asset strategies and multi-stock event studies

