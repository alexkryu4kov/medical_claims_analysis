from pathlib import Path

import pandas as pd

from preprocessor import ClaimsPreprocessor

m1 = pd.Timestamp("2020-05-01")
m2 = pd.Timestamp("2020-06-01")


def delta_by_dim(df, dim, m_prev, m_cur, top=15):
    sub = df[df["MONTH"].isin([m_prev, m_cur])]
    net = sub.groupby(["MONTH", dim])["PAID_AMOUNT"].sum().unstack(fill_value=0)
    # Свод: суммы по обоим месяцам и дельта (июнь - май)
    cur = net.loc[m_cur] if m_cur in net.index else pd.Series(0, index=net.columns)
    prev = net.loc[m_prev] if m_prev in net.index else pd.Series(0, index=net.columns)
    out = pd.DataFrame({"m_prev": prev, "m_cur": cur, "delta": cur - prev})
    return out.sort_values("delta").head(top), out.sort_values(
        "delta", ascending=False
    ).head(top)


def volume_price_breakdown(df, dim, m_prev, m_cur, top=15):
    pos = df[df["PAID_AMOUNT"] > 0].copy()
    g = (
        pos[pos["MONTH"].isin([m_prev, m_cur])]
        .groupby(["MONTH", dim])["PAID_AMOUNT"]
        .agg(sum="sum", count="size")
        .reset_index()
    )

    pivot_sum = g.pivot(index=dim, columns="MONTH", values="sum").fillna(0)
    pivot_count = g.pivot(index=dim, columns="MONTH", values="count").fillna(0)

    # Назовём колонки для удобства
    if m_prev not in pivot_sum.columns:
        pivot_sum[m_prev] = 0
    if m_cur not in pivot_sum.columns:
        pivot_sum[m_cur] = 0
    if m_prev not in pivot_count.columns:
        pivot_count[m_prev] = 0
    if m_cur not in pivot_count.columns:
        pivot_count[m_cur] = 0

    res = pd.DataFrame(
        {
            "sum_prev": pivot_sum[m_prev],
            "sum_cur": pivot_sum[m_cur],
            "rows_prev": pivot_count[m_prev],
            "rows_cur": pivot_count[m_cur],
        }
    )
    res["avg_prev"] = res["sum_prev"] / res["rows_prev"].replace(0, pd.NA)
    res["avg_cur"] = res["sum_cur"] / res["rows_cur"].replace(0, pd.NA)

    res["d_sum"] = res["sum_cur"] - res["sum_prev"]
    res["d_rows"] = res["rows_cur"] - res["rows_prev"]
    res["d_avg"] = res["avg_cur"] - res["avg_prev"]

    return res.sort_values("d_sum").head(top)


prep = ClaimsPreprocessor(Path("../claims_sample_data.csv")).load().preprocess()
df = prep.get_df()

cat_drop, cat_gain = delta_by_dim(df, "SERVICE_CATEGORY", m1, m2)
payer_drop, payer_gain = delta_by_dim(df, "PAYER", m1, m2)

print("\nTop категорий по падению NET (май→июнь):")
print(cat_drop.to_string())
print("\nTop плательщиков по падению NET (май→июнь):")
print(payer_drop.to_string())


print("\nГде просел объём/чек по категориям (топ падения gross):")
print(volume_price_breakdown(df, "SERVICE_CATEGORY", m1, m2).to_string())
print("\nГде просел объём/чек по плательщикам (топ падения gross):")
print(volume_price_breakdown(df, "PAYER", m1, m2).to_string())

focus_cats = cat_drop.index[:3].tolist()

for c in focus_cats:
    d = df[(df["SERVICE_CATEGORY"] == c) & (df["MONTH"].isin([m1, m2]))]
    by_spec = (
        d.groupby(["MONTH", "CLAIM_SPECIALTY"])["PAID_AMOUNT"]
        .sum()
        .unstack(fill_value=0)
    )
    if m1 not in by_spec.index:
        by_spec.loc[m1] = 0
    if m2 not in by_spec.index:
        by_spec.loc[m2] = 0
    delta = (by_spec.loc[m2] - by_spec.loc[m1]).sort_values()
    print(f"\n[{c}] Топ specialties по падению NET (май→июнь):")
    print(delta.head(15).to_string())
