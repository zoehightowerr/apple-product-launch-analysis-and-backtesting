from typing import List, Tuple, Dict
import pandas as pd
import numpy as np


def build_offset_grid(buy_offsets: List[int],
                      sell_offsets: List[int]) -> List[Tuple[int, int]]:
    """
    Return every (buy_offset, sell_offset) pair in Cartesian product.
    """
    return [(b, s) for b in buy_offsets for s in sell_offsets if s > b]


def adjusted_price(df: pd.DataFrame,
                   launch_date: pd.Timestamp,
                   offset: int) -> float:
    """
    Fetch the close price at the first trading day ≥ launch_date, then
    shift by `offset` trading days.
    """
    pos = df.index.get_indexer([launch_date], method='bfill')[0]
    if pos == -1:
        raise ValueError(f"Launch date {launch_date.date()} is outside data range")

    target_pos = pos + offset
    if not (0 <= target_pos < len(df)):
        raise ValueError(f"Offset {offset} for launch date "
                         f"{launch_date.date()} is out of bounds")

    return df.iloc[target_pos]["Close"]


def trade_return(df: pd.DataFrame,
                 launch_date: pd.Timestamp,
                 buy_offset: int,
                 sell_offset: int) -> float:
    """
    % return for buying at (launch + buy_offset) and selling at
    (launch + sell_offset).
    """
    buy = adjusted_price(df, launch_date, buy_offset)
    sell = adjusted_price(df, launch_date, sell_offset)
    return (sell - buy) / buy

def run_backtest(df_prices: pd.DataFrame,
                 launch_dates: List[pd.Timestamp],
                 offset_grid: List[Tuple[int, int]]
                 ) -> Dict[Tuple[int, int], List[float]]:
    """
    Loop through launches and offsets, returning a mapping
    {(buy_offset, sell_offset): [trade_returns…]}.
    """
    results: Dict[Tuple[int, int], List[float]] = {pair: [] for pair in offset_grid}

    for date in launch_dates:
        for pair in offset_grid:
            ret = trade_return(df_prices, date, *pair)
            results[pair].append(ret)

    return results


def aggregate_metrics(returns: List[float]) -> Dict[str, float]:
    """
    Calculate key stats for a list of trade returns.
    """
    arr = np.array(returns)
    mu = arr.mean()
    sigma = arr.std(ddof=1)
    sharpe = mu / sigma if sigma else np.nan
    hit_rate = (arr > 0).mean()
    return {"mean": mu, "std": sigma, "sharpe": sharpe, "hit_rate": hit_rate}

def summarize_backtest(backtest_results: Dict[Tuple[int, int], List[float]]
                        ) -> pd.DataFrame:
    """
    Turn the dict from run_backtest into a DataFrame:

    buy  sell  mean  std  sharpe  hit_rate
    -3    +5   …     …     …        …

    """
    rows = []
    for (b, s), rets in backtest_results.items():
        m = aggregate_metrics(rets)
        rows.append({"buy": b, "sell": s, **m})
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)

def backtest_per_product_type(
    prices: pd.DataFrame,
    launch_df: pd.DataFrame,
    buy_offsets: List[int],
    sell_offsets: List[int],
) -> pd.DataFrame:
    """
    Returns one big DataFrame with Sharpe metrics
    for every (product_type, buy_offset, sell_offset) combo.
    """
    grid = build_offset_grid(buy_offsets, sell_offsets)
    tables = [] 

    for ptype, grp in launch_df.groupby("Product Type"):
        dates   = grp["Release Date"].tolist()
        if len(dates) < 5: 
            print(f"⚠️  Skipping {ptype} (only {len(dates)} events)")
            continue

        results = run_backtest(prices, dates, grid)
        summary = summarize_backtest(results)
        summary["product_type"] = ptype
        summary["n_events"] = len(dates)
        tables.append(summary)

    if not tables:
        raise ValueError("No product types had enough events.")

    combo = pd.concat(tables, ignore_index=True)
    
    combo["rank_within_type"] = combo.groupby("product_type")["sharpe"].rank(
        method="dense", ascending=False
    )
    
    combo = combo.sort_values("sharpe", ascending=False)
    return combo
