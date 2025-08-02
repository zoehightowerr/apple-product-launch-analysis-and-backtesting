import numpy as np
from scipy.stats import ttest_1samp

from scripts.data_loader       import load_data
from scripts.event_study       import event_study
from scripts.strategy_backtest import (
    build_offset_grid,
    run_backtest,
    summarize_backtest,
    backtest_per_product_type
)
from scripts.monte_carlo       import (
    bootstrap_abnormal_returns,
    simulate_strategy,
    confidence_interval,
    block_bootstrap,
    probability_of_success
)
from scripts.risk_metrics      import calculate_risk_metrics_summary
from scripts.viz    import plot_sharpe_heatmap

if __name__ == "__main__":
    # 1) Load & prepare data
    aapl_prices, sp500_prices, launches = load_data()
    aapl_prices, sp500_prices, launches = event_study(
        aapl_prices, sp500_prices, launches
    )
    print("\n=== Top 5 Product Launches by CAR for Each Product Type ===")
    for ptype, group in launches.groupby("Product Type"):
        top5 = group.sort_values("CAR", ascending=False).head(5)
        print(f"\n▶ {ptype} (top 5 by CAR):")
        for _, row in top5.iterrows():
            print(f"  {row['Release Date'].date()} | {row['Family']} | CAR: {row['CAR']:.4f}")

    # 2) Define buy/sell offset ranges
    buy_offsets  = list(range(6))      # 0–5 days after release
    sell_offsets = list(range(1, 11))  # 1–10 days after release

    # 3) Backtest all combos and rank by Sharpe
    leaderboard = backtest_per_product_type(
        aapl_prices, launches, buy_offsets, sell_offsets
    )
    for p in leaderboard['product_type'].unique():
        plot_sharpe_heatmap(leaderboard, p)

    # 4) Print overall top 15 Sharpe ratios
    print("\n=== Best Sharpe ratios by product family ===")
    print(leaderboard.head(15).to_string(index=False))

    # 5) Extract the single best rule per product type
    best_per_type = (
        leaderboard
        .sort_values(["product_type", "sharpe"], ascending=[True, False])
        .groupby("product_type")
        .head(1)
    )
    print("\n=== Top rule for each product family ===")
    print(best_per_type[[
        "product_type", "buy", "sell", "sharpe", "hit_rate", "n_events"
    ]].to_string(index=False))

    # 6) Monte Carlo & risk metrics for each best rule
    print("\n=== Monte Carlo & Risk‐Metrics per product family ===")
    for _, row in best_per_type.iterrows():
        ptype = row["product_type"]
        buy   = int(row["buy"])
        sell  = int(row["sell"])

        # Monte Carlo on the abnormal‐return series
        ar = aapl_prices["Abnormal Return"].dropna()
        sims = bootstrap_abnormal_returns(ar)               # shape: (n_sims, len(ar))
        mc   = simulate_strategy(sims, buy, sell)           # dict of lists

        # convert lists to arrays so probability_of_success works
        sharpe_vals = np.array(mc["sharpe"])
        mean_vals   = np.array(mc["mean"])

        sharpe_ci = confidence_interval(sharpe_vals, 0.05, 0.95)
        prob_pos  = probability_of_success(mean_vals, 0)

        # Actual backtest returns for this (buy, sell) pair
        dates     = launches[launches["Product Type"] == ptype]["Release Date"].tolist()
        real_map  = run_backtest(aapl_prices, dates, [(buy, sell)])
        real_rets = real_map[(buy, sell)]
        risk_sum  = calculate_risk_metrics_summary(real_rets)

        # Print the summary
        print(f"\n→ {ptype}: buy @+{buy}, sell @+{sell}")
        print(f"   • MC Sharpe 90% CI : [{sharpe_ci[0]:.2f}, {sharpe_ci[1]:.2f}]")
        print(f"   • P(mean>0)        : {prob_pos:.1%}")
        print("   • Real risk summary:")
        for metric, value in risk_sum.items():
            print(f"       – {metric:15s}: {value:.4f}")

    
