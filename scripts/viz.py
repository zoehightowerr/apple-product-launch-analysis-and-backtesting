import seaborn as sns
import matplotlib.pyplot as plt

def plot_sharpe_heatmap(df, product_type):
    subset = df[df['product_type'] == product_type]
    pivot = subset.pivot(index="sell", columns="buy", values="sharpe")
    plt.figure(figsize=(8,6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title(f"Sharpe Ratio Heatmap: {product_type}")
    plt.xlabel("Buy Offset (days)")
    plt.ylabel("Sell Offset (days)")
    plt.tight_layout()
    plt.show()
