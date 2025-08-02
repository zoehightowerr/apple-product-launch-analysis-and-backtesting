import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def stock_collection(name: str) -> pd.DataFrame:
    """
    Downloads adjusted stock data for the given ticker between 2010 and May 2025.
    """
    data = yf.download(name, start="1980-12-12", end="2025-05-01", auto_adjust=True)
    return data[['Close']] 

def categorize(name: str) -> str:
        name = str(name).lower()
        if "iphone" in name: return "iPhone"
        if "mac" in name: return "Mac"
        if "ipod" in name: return "iPod"
        if "ipad" in name: return "iPad"
        if "headphones" in name: return "AirPods"
        else: return "Other"

def csv_to_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
    df = df.dropna(subset=['Release Date'])

    df['Product Type'] = df['Family'].apply(categorize)

    keep = {"iPhone", "iPad", "Mac", "AirPods", "iPod"}
    df = df[df['Product Type'].isin(keep)].copy()

    # Filter date range
    df = df[(df['Release Date'] >= '1980-12-12') & (df['Release Date'] <= '2025-05-01')]

    if 'Discontinued Date' in df.columns:
        df['Discontinued Date'] = pd.to_datetime(df['Discontinued Date'], errors='coerce')
        df['Cycle Length'] = (df['Discontinued Date'] - df['Release Date']).dt.days

    return df


def clean_stock_data(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Forward fills missing values and shifts holidays/weekends to the previous trading day.
    Assumes index is datetime.
    """
    stock_data = stock_data.copy()
    stock_data.ffill(inplace=True)
    stock_data.index = pd.to_datetime(stock_data.index)
    return stock_data

def load_data():
    # Collect stock data
    aapl = clean_stock_data(stock_collection("AAPL"))
    sp500 = clean_stock_data(stock_collection("^GSPC"))
    aapl_launch = csv_to_df("/Users/zoehightower/apple-product-launch-analysis-and-backtesting/data/APPLE_DATASET.csv")

    return aapl,sp500,aapl_launch

