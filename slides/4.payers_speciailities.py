# service_specialty_mix_top_payers.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from preprocessor import ClaimsPreprocessor

# ---------- utils ----------
def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)

def build_combo(df: pd.DataFrame, sep: str = " · ") -> pd.Series:
    sc = df["SERVICE_CATEGORY"].astype(str).str.strip()
    sp = df["CLAIM_SPECIALTY"].astype(str).str.strip()
    # заменим пустые/NaN-подобные на 'Unknown'
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
    return (d.groupby("PAYER")["net"].sum()
             .sort_values(ascending=False).head(k).index.tolist())

def combo_values_for_payer(df: pd.DataFrame, payer: str, value: str = "net",
                           top_n: int = 8, skip_last: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Возвращает:
      values_top (MONTH × (topN + Other)) — суммы value ('net'/'pos')
      shares_top — доли (0..1) этих же столбцов внутри месяца
    """
    assert value in {"net","pos"}
    d = df[df["PAYER"] == payer].copy()
    amt = pd.to_numeric(d["PAID_AMOUNT"], errors="coerce")
    d["pos"] = amt.where(amt > 0, 0.0)
    d["neg"] = -amt.where(amt < 0, 0.0)
    d["net"] = d["pos"] - d["neg"]
    d["COMBO"] = build_combo(d)

    # MONTH × COMBO
    pivot = (d.groupby(["MONTH","COMBO"])[value]
               .sum().unstack("COMBO", fill_value=0.0)
               .sort_index())

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

# ---------- plotting ----------
def plot_values_stack(vals: pd.DataFrame, title: str, outfile: str):
    idx = vals.index
    x = np.arange(len(idx))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(x, vals.T.values, labels=vals.columns.tolist())
    ax.set_title(title)
    ax.set_xlabel("Месяц")
    ax.set_ylabel("Сумма (NET)")
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%Y-%m") for d in idx], rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=min(4, len(vals.columns)), loc="upper center", bbox_to_anchor=(0.5, -0.15))
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close(fig)

def plot_shares_stack(shares: pd.DataFrame, title: str, outfile: str, highlight_month: pd.Timestamp | None = None):
    idx = shares.index
    x = np.arange(len(idx))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(x, shares.T.values, labels=shares.columns.tolist())
    ax.set_title(title)
    ax.set_xlabel("Месяц")
    ax.set_ylabel("Доля")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%Y-%m") for d in idx], rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    if highlight_month is not None and highlight_month in idx:
        ix = np.where(idx == highlight_month)[0]
        if len(ix): ax.axvline(ix[0], color="k", linestyle="--", linewidth=1)
    ax.legend(ncol=min(4, len(shares.columns)), loc="upper center", bbox_to_anchor=(0.5, -0.15))
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close(fig)

def plot_delta_bar(vals: pd.DataFrame, m_prev: pd.Timestamp, m_cur: pd.Timestamp, title: str, outfile: str):
    need = [m for m in (m_prev, m_cur) if m in vals.index]
    if len(need) < 2:
        print(f"[warn] Нет обоих месяцев для delta: {m_prev.date()} / {m_cur.date()}")
        return
    dlt = (vals.loc[m_cur] - vals.loc[m_prev]).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["tab:red" if v < 0 else "tab:green" for v in dlt.values]
    ax.barh(dlt.index, dlt.values, color=colors)
    ax.set_title(title)
    ax.set_xlabel("Δ NET (June - May)")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close(fig)

# ---------- main ----------
def main(input_path: str = "claims_sample_data.csv",
         outdir: str = "out",
         k_top_payers: int = 3,
         top_n_combos: int = 15,
         value: str = "net",
         skip_last: bool = True):

    Path(outdir).mkdir(parents=True, exist_ok=True)

    # 1) загрузка и препроцессинг (исключение PCPEncounter уже внутри preprocessor)
    df = ClaimsPreprocessor(Path(input_path)).load().preprocess().df

    # 2) топ плательщики (по NET за 12 мес)
    top_payers = get_top_payers(df, k=k_top_payers, last_n_months=12)
    print("Top payers:", ", ".join(top_payers))

    # 3) месяцы для подсветки/дельты
    months_sorted = sorted(df["MONTH"].unique())
    month_june = pd.Timestamp("2020-06-01") if pd.Timestamp("2020-06-01") in months_sorted else None
    month_may  = pd.Timestamp("2020-05-01") if pd.Timestamp("2020-05-01") in months_sorted else (months_sorted[-3] if len(months_sorted) >= 3 else None)

    # 4) по каждому payer строим графики по COMBO
    for p in top_payers:
        vals, shares = combo_values_for_payer(df, payer=p, value=value, top_n=top_n_combos, skip_last=skip_last)
        if vals.empty:
            print(f"[skip] {p}: нет данных")
            continue

        base = sanitize(p)
        plot_values_stack(
            vals,
            title=f"{p} — NET по топ-{top_n_combos} COMBO (SERVICE · SPECIALTY)",
            outfile=str(Path(outdir) / f"{base}_combo_values.png")
        )
        plot_shares_stack(
            shares,
            title=f"{p} — Доли COMBO в NET (SERVICE · SPECIALTY)",
            outfile=str(Path(outdir) / f"{base}_combo_shares.png"),
            highlight_month=month_june
        )

        # вклад COMBO в ∆ Май→Июнь
        if month_may is not None and month_june is not None:
            plot_delta_bar(
                vals, m_prev=month_may, m_cur=month_june,
                title=f"{p} — What Do Top Payers Pay For?",
                outfile=str(Path(outdir) / f"{base}_combo_delta_May_to_June.png")
            )

        # контрольная распечатка (можно убрать)
        print(f"\n[{p}] последние 12 мес — доли топ COMBO, %:")
        print((shares.tail(12) * 100).round(1).to_string())

if __name__ == "__main__":
    main()
