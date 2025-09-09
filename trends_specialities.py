from pathlib import Path

import numpy as np
import pandas as pd

from preprocessor import ClaimsPreprocessor

prep = ClaimsPreprocessor(Path("claims_sample_data.csv")).load().preprocess()
df = prep.get_df()


def top_specialty_monthly(
    df: pd.DataFrame,
    top_n: int = 10,
    value: str = "net",
    last_n_months: int | None = None,
) -> pd.DataFrame:
    assert value in {"net", "pos"}

    d = df.copy()

    amt = pd.to_numeric(d["PAID_AMOUNT"], errors="coerce")
    d["pos"] = amt.where(amt > 0, 0.0)
    d["neg"] = -amt.where(amt < 0, 0.0)
    d["net"] = d["pos"] - d["neg"]

    months = sorted(d["MONTH"].unique())
    if last_n_months and len(months) > last_n_months:
        d = d[d["MONTH"] >= months[-last_n_months]]

    top_specs = (
        d.groupby("CLAIM_SPECIALTY")[value]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )

    pivot = (
        d.groupby(["MONTH", "CLAIM_SPECIALTY"])[value]
        .sum()
        .unstack("CLAIM_SPECIALTY", fill_value=0.0)
        .sort_index()
    )

    vals = pd.DataFrame(index=pivot.index)
    for c in top_specs:
        if c in pivot.columns:
            vals[c] = pivot[c]
        else:
            vals[c] = 0.0

    other_cols = [c for c in pivot.columns if c not in top_specs]
    if other_cols:
        vals["Other"] = pivot[other_cols].sum(axis=1)

    order = (
        vals.drop(columns=["Other"], errors="ignore")
        .sum(axis=0)
        .sort_values(ascending=False)
        .index.tolist()
    )
    if "Other" in vals.columns:
        order += ["Other"]
    vals = vals.reindex(columns=order)

    return vals


def top_specialty_shares(monthly_vals: pd.DataFrame) -> pd.DataFrame:
    totals = monthly_vals.sum(axis=1).replace(0, np.nan)
    return monthly_vals.div(totals, axis=0)


monthly_top = top_specialty_monthly(
    df,
    top_n=6,
    value="net",
    last_n_months=12,
)

print("Топ-6 SPECIALTY по NET, помесячные суммы (последние 12 мес):")
print(monthly_top.tail(12).round(0).astype(int).to_string())

shares_top = top_specialty_shares(monthly_top)
print("\nДоли этих SPECIALTY (%, последние 12 мес):")
print((shares_top.tail(12) * 100).round(1).to_string())

mom_top = monthly_top.pct_change().replace([np.inf, -np.inf], np.nan)
print("\nMoM по SPECIALTY (в долях, последние 12 мес):")
print(mom_top.tail(12).round(3).to_string())
