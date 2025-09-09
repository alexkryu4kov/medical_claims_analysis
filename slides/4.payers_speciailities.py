from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocessor import ClaimsPreprocessor


def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


def build_combo(df: pd.DataFrame, sep: str = " · ") -> pd.Series:
    sc = df["SERVICE_CATEGORY"].astype(str).str.strip()
    sp = df["CLAIM_SPECIALTY"].astype(str).str.strip()
    sc = sc.replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
    sp = sp.replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
    return (sc + sep + sp).rename("COMBO")


def get_top_payers(df: pd.DataFrame, k: int = 3, last_n_months: int = 12) -> list[str]:
    amt = pd.to_numeric(df["PAID_AMOUNT"], errors="coerce")
    d = df.assign(pos=amt.where(amt > 0, 0.0), neg=-amt.where(amt < 0, 0.0))
    d["net"] = d["pos"] - d["neg"]
    months = sorted(d["MONTH"].unique())
    if last_n_months and len(months) > last_n_months:
        d = d[d["MONTH"] >= months[-last_n_months]]
    return (
        d.groupby("PAYER")["net"]
        .sum()
        .sort_values(ascending=False)
        .head(k)
        .index.tolist()
    )


def combo_values_for_payer(
    df: pd.DataFrame,
    payer: str,
    value: str = "net",
    top_n: int = 8,
    skip_last: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df[df["PAYER"] == payer].copy()
    amt = pd.to_numeric(d["PAID_AMOUNT"], errors="coerce")
    d["pos"] = amt.where(amt > 0, 0.0)
    d["neg"] = -amt.where(amt < 0, 0.0)
    d["net"] = d["pos"] - d["neg"]
    d["COMBO"] = build_combo(d)

    # MONTH × COMBO
    pivot = (
        d.groupby(["MONTH", "COMBO"])[value]
        .sum()
        .unstack("COMBO", fill_value=0.0)
        .sort_index()
    )

    if skip_last and len(pivot) > 0:
        pivot = pivot.iloc[:-1]

    totals = pivot.sum(axis=0).sort_values(ascending=False)
    top_cols = list(totals.index[:top_n])

    vals = pivot[top_cols].copy()
    if pivot.shape[1] > top_n:
        vals["Other"] = pivot.drop(columns=top_cols).sum(axis=1)

    totals_row = vals.sum(axis=1).replace(0, np.nan)
    shares = vals.div(totals_row, axis=0)
    return vals, shares


def plot_delta_bar(
    vals: pd.DataFrame,
    m_prev: pd.Timestamp,
    m_cur: pd.Timestamp,
    title: str,
    outfile: str,
):
    need = [m for m in (m_prev, m_cur) if m in vals.index]
    if len(need) < 2:
        print(f"[warn] Нет обоих месяцев для delta: {m_prev.date()} / {m_cur.date()}")
        return
    dlt = (vals.loc[m_cur] - vals.loc[m_prev]).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["tab:red" if v < 0 else "tab:green" for v in dlt.values]
    ax.barh(dlt.index, dlt.values, color=colors)
    ax.set_title(title)
    ax.set_xlabel("NET (June - May)")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close(fig)


def main(
    input_path: str = "claims_sample_data.csv",
    outdir: str = "out",
    k_top_payers: int = 3,
    top_n_combos: int = 15,
    value: str = "net",
    skip_last: bool = True,
):

    Path(outdir).mkdir(parents=True, exist_ok=True)

    df = ClaimsPreprocessor(Path(input_path)).load().preprocess().df

    top_payers = get_top_payers(df, k=k_top_payers, last_n_months=12)
    print("Top payers:", ", ".join(top_payers))

    months_sorted = sorted(df["MONTH"].unique())
    month_june = (
        pd.Timestamp("2020-06-01")
        if pd.Timestamp("2020-06-01") in months_sorted
        else None
    )
    month_may = (
        pd.Timestamp("2020-05-01")
        if pd.Timestamp("2020-05-01") in months_sorted
        else (months_sorted[-3] if len(months_sorted) >= 3 else None)
    )

    for p in top_payers:
        vals, shares = combo_values_for_payer(
            df, payer=p, value=value, top_n=top_n_combos, skip_last=skip_last
        )
        if vals.empty:
            print(f"[skip] {p}: нет данных")
            continue

        base = sanitize(p)

        if month_may is not None and month_june is not None:
            plot_delta_bar(
                vals,
                m_prev=month_may,
                m_cur=month_june,
                title=f"{p} — What Do Top Payers Pay For?",
                outfile=str(Path(outdir) / f"{base}_combo_delta_May_to_June.png"),
            )


if __name__ == "__main__":
    main()
