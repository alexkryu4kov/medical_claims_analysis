# top3_payers_three_plots_nomom.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from preprocessor import ClaimsPreprocessor

def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)

def top_payers_by_net(df: pd.DataFrame, k: int = 3, last_n_months: int = 12) -> list[str]:
    amt = pd.to_numeric(df["PAID_AMOUNT"], errors="coerce")
    d = df.assign(pos=amt.where(amt > 0, 0.0), neg=-amt.where(amt < 0, 0.0))
    d["net"] = d["pos"] - d["neg"]
    months = sorted(d["MONTH"].unique())
    if last_n_months and len(months) > last_n_months:
        d = d[d["MONTH"] >= months[-last_n_months]]
    return (d.groupby("PAYER")["net"].sum()
              .sort_values(ascending=False).head(k).index.tolist())

def monthly_net_by_payer(df: pd.DataFrame) -> pd.DataFrame:
    amt = pd.to_numeric(df["PAID_AMOUNT"], errors="coerce")
    d = df.assign(pos=amt.where(amt > 0, 0.0), neg=-amt.where(amt < 0, 0.0))
    d["net"] = d["pos"] - d["neg"]
    return (d.groupby(["MONTH","PAYER"])["net"]
              .sum().unstack("PAYER", fill_value=0.0).sort_index())

def plot_one_payer(x_idx: pd.Index, y: pd.Series, outfile: str, title_prefix: str = ""):
    ma3 = y.rolling(3, min_periods=1).mean()
    mom = y.pct_change().replace([np.inf, -np.inf], np.nan)  # только для метрики стабильности
    med_abs_mom = mom.dropna().tail(12).abs().median()
    label = "Stable" if pd.notna(med_abs_mom) and med_abs_mom < 0.10 else "Volatile"
    med_txt = "—" if pd.isna(med_abs_mom) else f"{med_abs_mom*100:.1f}%"

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_idx, y, marker="o", linewidth=1.8, label="NET")
    ax.plot(x_idx, ma3, marker="o", linewidth=1.8, label="MA3")

    ax.set_title(f"{title_prefix} — {label} (median |MoM|={med_txt})")
    ax.set_xlabel("Month")
    ax.set_ylabel("NET")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xticks(x_idx)
    ax.set_xticklabels([d.strftime("%Y-%m") for d in x_idx], rotation=45, ha="right")
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close(fig)
    print(f"Saved: {outfile}")

def main(input_path: str = "claims_sample_data.csv", skip_last: bool = True):
    df = ClaimsPreprocessor(Path(input_path)).load().preprocess().df  # PCPEncounter уже фильтруется в препроцессоре
    top3 = top_payers_by_net(df, k=3, last_n_months=12)
    panel = monthly_net_by_payer(df)[top3]

    if skip_last and len(panel) > 0:
        panel = panel.iloc[:-1]

    for p in top3:
        s = panel[p]
        if s.dropna().shape[0] < 2:
            print(f"[skip] {p}: слишком мало точек для графика")
            continue
        fname = f"payer_trend_{sanitize(p)}.png"
        plot_one_payer(panel.index, s, outfile=fname, title_prefix=p)

if __name__ == "__main__":
    main()
