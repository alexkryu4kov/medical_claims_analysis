from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from preprocessor import ClaimsPreprocessor


def payer_shares_over_time(df: pd.DataFrame, value: str = "net", top_n: int = 8):
    amt = pd.to_numeric(df["PAID_AMOUNT"], errors="coerce")
    tmp = df.assign(pos=amt.where(amt > 0, 0.0), neg=-amt.where(amt < 0, 0.0))
    by = (
        tmp.groupby(["MONTH", "PAYER"])
        .agg(pos=("pos", "sum"), neg=("neg", "sum"))
        .sort_index()
    )
    by["net"] = by["pos"] - by["neg"]
    assert value in {"net", "pos"}
    values = by[value].unstack("PAYER", fill_value=0.0).sort_index()
    totals = values.sum(axis=1).replace(0, np.nan)
    shares = values.div(totals, axis=0)
    total_by_payer = values.sum(axis=0).sort_values(ascending=False)
    top = list(total_by_payer.index[:top_n])
    shares_top = shares[top].copy()
    if shares.shape[1] > top_n:
        shares_top["Other"] = shares.drop(columns=top).sum(axis=1)
    return values, shares, shares_top


def plot_payer_share_stacked(
    df: pd.DataFrame,
    top_n: int = 8,
    value: str = "net",
    skip_last: bool = True,
    outfile: str = "payer_share_stacked.png",
):
    _, _, shares_top = payer_shares_over_time(df, value=value, top_n=top_n)

    if skip_last and len(shares_top) > 0:
        shares_top = shares_top.iloc[:-1]

    # stackplot
    idx = shares_top.index
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(idx, shares_top.T.values)
    ax.set_title(f"Top Payers Over Time (NET)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Share")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(
        shares_top.columns.tolist(),
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
    )
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    print(f"Saved: {outfile}")


prep = ClaimsPreprocessor(Path("../claims_sample_data.csv")).load().preprocess()
df = prep.get_df()


plot_payer_share_stacked(
    df, top_n=8, value="net", skip_last=True, outfile="payer_share_stacked.png"
)
