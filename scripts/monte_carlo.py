import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp


def bootstrap_abnormal_returns(ar: pd.Series, n_simulations: int = 10_000) -> np.ndarray:
    """Generate bootstrapped abnormal-return series."""
    n = len(ar)
    sims = np.random.choice(ar.dropna(), size=(n_simulations, n), replace=True)
    return sims


def simulate_strategy(sims: np.ndarray, buy_offset: int, sell_offset: int) -> dict:
    """Compute returns, Sharpe, hit rate for each simulated path."""
    results = {'mean': [], 'std': [], 'sharpe': [], 'hit_rate': []}
    for seq in sims:
        ret = (seq[sell_offset:] - seq[buy_offset:-sell_offset+buy_offset]) / seq[buy_offset:-sell_offset+buy_offset]
        mu, sigma = ret.mean(), ret.std(ddof=1)
        results['mean'].append(mu)
        results['std'].append(sigma)
        results['sharpe'].append(mu / sigma if sigma else np.nan)
        results['hit_rate'].append((ret > 0).mean())
    return results


def confidence_interval(data: np.ndarray, lower: float = 0.05, upper: float = 0.95) -> tuple:
    """Return (lower, upper) percentiles of data."""
    return np.percentile(data, [100 * lower, 100 * upper])


def t_test_cars(cars: pd.Series, alpha: float = 0.05) -> dict:
    """One-sample t-test of CARs vs zero."""
    t, p = ttest_1samp(cars.dropna(), popmean=0)
    return {'t_stat': t, 'p_val': p, 'significant': p < alpha}


def block_bootstrap(ar: pd.Series, block_size: int = 3, n_simulations: int = 10_000) -> np.ndarray:
    """Resample returns in blocks to preserve clustering."""
    n = len(ar)
    sims = []
    for _ in range(n_simulations):
        idx = np.random.randint(0, n - block_size + 1, size=(n // block_size) + 1)
        blocks = [ar.iloc[i:i + block_size].values for i in idx]
        sims.append(np.concatenate(blocks)[:n])
    return np.array(sims)


def probability_of_success(metrics: np.ndarray, threshold: float) -> float:
    """Fraction of simulations where metric exceeds threshold."""
    return np.mean(metrics > threshold)


