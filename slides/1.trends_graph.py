from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import StrMethodFormatter

from preprocessor import ClaimsPreprocessor

prep = ClaimsPreprocessor(Path("../claims_sample_data.csv")).load().preprocess()
df = prep.get_df()

amt = pd.to_numeric(df["PAID_AMOUNT"], errors="coerce")
tmp = df.assign(pos=amt.where(amt > 0, 0.0), neg=-amt.where(amt < 0, 0.0))
monthly = (
    tmp.groupby("MONTH")[["pos", "neg"]]
    .sum()
    .sort_index()
    .assign(net=lambda x: x["pos"] - x["neg"])
)
monthly["net_MA3"] = monthly["net"].rolling(3, min_periods=1).mean()

monthly = monthly.sort_index()

# Remove July (last month) cause data is insufficient
if len(monthly) > 0:
    monthly_plot = monthly.iloc[:-1].copy()
else:
    monthly_plot = monthly.copy()

print(monthly_plot.tail())

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(monthly_plot.index, monthly_plot["net"], marker="o", label="NET (monthly)")
ax.plot(monthly_plot.index, monthly_plot["net_MA3"], marker="o", label="NET (3-mo MA)")

ax.set_title("Monthly Net Revenue Trend")
ax.set_xlabel("Month")
ax.set_ylabel("Amount")
ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
for lbl in ax.get_xticklabels():
    lbl.set_rotation(45)
    lbl.set_ha("right")
ax.grid(True, alpha=0.3)
ax.legend()

if not monthly_plot.empty:
    ax.set_xlim(monthly_plot.index.min(), monthly_plot.index.max())

plt.tight_layout()
plt.savefig("trend.png", dpi=200)
print("Saved: trend.png")

print("\nLast 12 months (pos/neg/net/MA3):")
print(
    monthly_plot[["pos", "neg", "net", "net_MA3"]]
    .tail(12)
    .round(0)
    .astype(int)
    .to_string()
)
