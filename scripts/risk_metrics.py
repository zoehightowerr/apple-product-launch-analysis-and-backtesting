import numpy as np


def calculate_value_at_risk(returns, alpha=0.05):
    """Value at Risk (alpha quantile of losses)."""
    if len(returns) == 0:
        return np.nan
    return np.percentile(returns, 100 * alpha)


def calculate_conditional_var(returns, alpha=0.05):
    """Average loss beyond VaR."""
    var = calculate_value_at_risk(returns, alpha)
    tail = [r for r in returns if r <= var]
    return np.mean(tail) if tail else var


def calculate_drawdown_series(returns):
    """Drawdown at each time from peak of cumulative returns."""
    cumulative = np.cumprod(1 + np.array(returns))
    peak = np.maximum.accumulate(cumulative)
    return (cumulative - peak) / peak


def calculate_maximum_drawdown(returns):
    """Max peak-to-trough decline and its duration."""
    dd = calculate_drawdown_series(returns)
    max_dd = dd.min()
    end = np.argmin(dd)
    start = np.argmax(np.maximum.accumulate(calculate_drawdown_series(returns)[:end])) if end > 0 else 0
    duration = end - start
    return {"max_drawdown": max_dd, "duration": duration}


def calculate_risk_metrics_summary(returns):
    """Summary of risk metrics: VaR, CVaR, drawdown, and more."""
    ret = np.array(returns)
    summary = {
        "VaR(5%)": calculate_value_at_risk(ret, 0.05),
        "CVaR(5%)": calculate_conditional_var(ret, 0.05),
        **calculate_maximum_drawdown(ret),
        "downside_deviation": np.std(ret[ret < 0], ddof=1) if any(ret < 0) else 0,
        "worst_loss": ret.min() if ret.size else np.nan,
        "loss_rate": np.mean(ret < 0) if ret.size else np.nan
    }
    # Sortino ratio = mean / downside deviation
    dd = summary["downside_deviation"]
    summary["Sortino"] = (ret.mean() / dd) if dd else np.nan
    return summary


def stress_test_strategy(returns, shock_factors=None):
    """Apply shocks to returns and compare summaries."""
    if shock_factors is None:
        shock_factors = {"x2_volatility": 2.0}
    results = {}
    for name, factor in shock_factors.items():
        shocked = returns * factor
        results[name] = calculate_risk_metrics_summary(shocked)
    return results
