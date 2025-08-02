import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import ttest_1samp

def simple_daily(stock_data: pd.DataFrame) -> pd.DataFrame:
    #calculate simple daily return
    stock_data['Simple Return'] = stock_data['Close'].pct_change()
    return stock_data

def log_daily(stock_data: pd.DataFrame) -> pd.DataFrame:
    #calculate log daily return
    stock_data['Log Return'] = np.log(stock_data.Close/stock_data.Close.shift(1))
    return stock_data

def market_relationship(apple_data: pd.DataFrame, sp500_data: pd.DataFrame) -> pd.DataFrame:
    # Ensure datetime index
    apple_data = apple_data.copy()
    sp500_data = sp500_data.copy()
    apple_data.index = pd.to_datetime(apple_data.index)
    sp500_data.index = pd.to_datetime(sp500_data.index)

    # Merge data on date
    combined = pd.DataFrame({
        'apple': apple_data['Log Return'],
        'market': sp500_data['Log Return']
    }).dropna()

    # Add year column
    combined['year'] = combined.index.year
    expected_returns = pd.Series(index=combined.index, dtype=float)

    for year, group in combined.groupby('year'):
        y = group['apple']
        X = sm.add_constant(group['market'])

        try:
            model = sm.OLS(y, X).fit()
            alpha = model.params['const']
            beta = model.params['market']
            expected = alpha + beta * group['market']
            expected_returns.loc[group.index] = expected
        except:
            # In case of failure (e.g., not enough data), fill NaN
            expected_returns.loc[group.index] = np.nan

    # Assign expected returns back to original DataFrame
    apple_data['Expected Return'] = expected_returns

    return apple_data


def abnormal_returns(apple_data: pd.DataFrame) -> pd.DataFrame:
    # calculate the difference between what actually happened vs expected
    apple_data['Abnormal Return'] = apple_data['Log Return']- apple_data['Expected Return']
    return apple_data

def cumulative_abnormal_returns(apple_data: pd.DataFrame, aapl_launch: pd.DataFrame) -> pd.DataFrame:
    #calculate the CAR

    car_values = []

    for date in aapl_launch['Release Date']:
        start = date - pd.Timedelta(days=5)
        end = date + pd.Timedelta(days=5)

        # Sum abnormal returns in this window
        window_returns = apple_data.loc[start:end, 'Abnormal Return']
        car = window_returns.sum()

        car_values.append(car)

    # Add to launch DataFrame
    aapl_launch['CAR'] = car_values


    return aapl_launch


def event_study(apple_data: pd.DataFrame, sp500_data: pd.DataFrame, aapl_launch: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    apple_data = log_daily(simple_daily(apple_data))
    sp500_data = log_daily(simple_daily(sp500_data))

    apple_data = market_relationship(apple_data, sp500_data)
    apple_data = abnormal_returns(apple_data)
    aapl_launch = cumulative_abnormal_returns(apple_data, aapl_launch)


    return apple_data, sp500_data, aapl_launch