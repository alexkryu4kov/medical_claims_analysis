from pathlib import Path

import numpy as np
import pandas as pd

from preprocessor import ClaimsPreprocessor

prep = ClaimsPreprocessor(Path("claims_sample_data.csv")).load().preprocess()
df = prep.get_df()


def top_specialty_monthly(
    df: pd.DataFrame,
    top_n: int = 10,
    value: str = "net",  # "net" (по умолчанию) или "pos"
    last_n_months: int | None = None,
    exclude_categories=("PCPEncounter",),
    drop_unknown: bool = True,
) -> pd.DataFrame:
    """
    Возвращает таблицу MONTH × (top-N specialties + Other) со значениями 'net' (или 'pos').
    'net' = pos - neg; pos = сумма положительных.
    """
    assert value in {"net", "pos"}

    d = df.copy()
    if exclude_categories:
        d = d[~d["SERVICE_CATEGORY"].isin(exclude_categories)]
    if drop_unknown and "CLAIM_SPECIALTY" in d.columns:
        d = d[d["CLAIM_SPECIALTY"].astype(str).str.strip().str.lower() != "unknown"]

    amt = pd.to_numeric(d["PAID_AMOUNT"], errors="coerce")
    d["pos"] = amt.where(amt > 0, 0.0)
    d["neg"] = -amt.where(amt < 0, 0.0)
    d["net"] = d["pos"] - d["neg"]

    # ограничим период при необходимости
    months = sorted(d["MONTH"].unique())
    if last_n_months and len(months) > last_n_months:
        d = d[d["MONTH"] >= months[-last_n_months]]

    # топ-N специализаций по выбранной метрике за период
    top_specs = (
        d.groupby("CLAIM_SPECIALTY")[value]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )

    # помесячные суммы по специализациям
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

    # «хвост» в Other (если есть не-топовые столбцы)
    other_cols = [c for c in pivot.columns if c not in top_specs]
    if other_cols:
        vals["Other"] = pivot[other_cols].sum(axis=1)

    # аккуратный порядок столбцов: топ по сумме за период + Other в конце
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
    """Доли (0..1) top-N специализаций и Other в каждом месяце."""
    totals = monthly_vals.sum(axis=1).replace(0, np.nan)
    return monthly_vals.div(totals, axis=0)


# ===== пример использования =====
# топ-6 специализаций по NET за последние 12 мес (без PCPEncounter)
monthly_top = top_specialty_monthly(
    df,
    top_n=6,
    value="net",
    last_n_months=12,
    exclude_categories=("PCPEncounter",),
    drop_unknown=True,
)

print("Топ-6 SPECIALTY по NET, помесячные суммы (последние 12 мес):")
print(monthly_top.tail(12).round(0).astype(int).to_string())

# при желании — доли (в %)
shares_top = top_specialty_shares(monthly_top)
print("\nДоли этих SPECIALTY (%, последние 12 мес):")
print((shares_top.tail(12) * 100).round(1).to_string())

# опционально — MoM по каждой специализации (для стабильности)
mom_top = monthly_top.pct_change().replace([np.inf, -np.inf], np.nan)
print("\nMoM по SPECIALTY (в долях, последние 12 мес):")
print(mom_top.tail(12).round(3).to_string())
