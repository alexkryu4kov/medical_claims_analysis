from pathlib import Path
from preprocessor import ClaimsPreprocessor
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

prep = ClaimsPreprocessor(Path("../claims_sample_data.csv")).load().preprocess()
df = prep.get_df()

amt = pd.to_numeric(df["PAID_AMOUNT"], errors="coerce")
tmp = df.assign(pos=amt.where(amt > 0, 0.0),
                neg=-amt.where(amt < 0, 0.0))
monthly = (tmp.groupby("MONTH")[["pos", "neg"]].sum()
             .sort_index()
             .assign(net=lambda x: x["pos"] - x["neg"]))
monthly["net_MA3"] = monthly["net"].rolling(3, min_periods=1).mean()

monthly = monthly.sort_index()

# 1) убираем последний месяц независимо от индекса
if len(monthly) > 0:
    monthly_plot = monthly.iloc[:-1].copy()
else:
    monthly_plot = monthly.copy()

print(monthly_plot.tail())

# 2) график
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(monthly_plot.index, monthly_plot["net"], marker="o", label="NET (monthly)")
ax.plot(monthly_plot.index, monthly_plot["net_MA3"], marker="o", label="NET (3-mo MA)")

ax.set_title("Monthly Net Revenue Trend")
ax.set_xlabel("Month")
ax.set_ylabel("Amount")
ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
for lbl in ax.get_xticklabels():
    lbl.set_rotation(45); lbl.set_ha("right")
ax.grid(True, alpha=0.3)
ax.legend()

# ВАЖНО: фиксируем пределы оси X, чтобы не было «лишних» тиков за пределами данных
if not monthly_plot.empty:
    ax.set_xlim(monthly_plot.index.min(), monthly_plot.index.max())

plt.tight_layout()
plt.savefig("trend.png", dpi=200)
print("Saved: trend.png")

print("\nLast 12 months (pos/neg/net/MA3):")
print(monthly_plot[["pos","neg","net","net_MA3"]].tail(12).round(0).astype(int).to_string())