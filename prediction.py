from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthBegin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessor import ClaimsPreprocessor

DROP_THRESH = 0.30  # threshold to predict


def monthly_net_by_payer(df: pd.DataFrame) -> pd.DataFrame:
    """MONTH × PAYER → NET (pos - neg)."""
    amt = pd.to_numeric(df["PAID_AMOUNT"], errors="coerce")
    d = df.assign(pos=amt.where(amt > 0, 0.0), neg=-amt.where(amt < 0, 0.0))
    d["net"] = d["pos"] - d["neg"]
    return (
        d.groupby(["MONTH", "PAYER"])["net"]
        .sum()
        .unstack("PAYER", fill_value=0.0)
        .sort_index()
    )


def build_features_for_payer(
    panel: pd.DataFrame, payer: str, drop_thresh: float = DROP_THRESH
):
    if payer not in panel.columns:
        raise ValueError(f"Плательщик '{payer}' не найден.")

    s = panel[payer]  # NET_t для payer
    total = panel.sum(axis=1).replace(0, np.nan)
    share_t = (s / total).rename("share_t")
    hhi_t = (panel.div(total, axis=0).pow(2).sum(axis=1)).rename("HHI_t")

    feat = pd.DataFrame(
        {
            "NET_t": s,
            "NET_lag1": s.shift(1),
            "MoM_t": s.pct_change().replace([np.inf, -np.inf], np.nan),
            "MA3_t": s.rolling(3, min_periods=1).mean(),
            "share_t": share_t,
            "HHI_t": hhi_t,
        }
    )

    m = feat.index.month
    feat["sin_m"] = np.sin(2 * np.pi * m / 12)
    feat["cos_m"] = np.cos(2 * np.pi * m / 12)

    y_next = (s.pct_change().shift(-1) < -drop_thresh).astype(float)

    mask_train = y_next.notna() & feat.notna().all(axis=1)
    X_train = feat[mask_train]
    y_train = y_next[mask_train]

    X_pred = feat.iloc[[-1]].dropna(axis=1)
    common_cols = X_train.columns.intersection(X_pred.columns)
    X_train = X_train[common_cols]
    X_pred = X_pred[common_cols]

    pred_month = (feat.index.max() + MonthBegin(1)).date()
    return X_train, y_train, X_pred, pred_month


def predict_drop_probability_next_month(
    df: pd.DataFrame, payer: str
) -> tuple[float, str]:
    panel = monthly_net_by_payer(df)
    X_train, y_train, X_pred, pred_month = build_features_for_payer(
        panel, payer, DROP_THRESH
    )

    if len(X_train) < 8:
        raise RuntimeError(
            f"Слишком мало данных для обучения по '{payer}' ({len(X_train)} точек)."
        )

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=42
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)

    prob = float(pipe.predict_proba(X_pred)[:, 1][0])
    return prob, pred_month


if __name__ == "__main__":
    df = ClaimsPreprocessor(Path("claims_sample_data.csv")).load().preprocess().get_df()

    payers_to_check = ["Payer F", "Payer H", "Payer CA"]

    results = []
    for name in payers_to_check:
        try:
            prob, month_date = predict_drop_probability_next_month(df, name)
            print(
                f"{name}: вероятность падения >{int(DROP_THRESH * 100)}% в {month_date}: {prob:.2%}"
            )
            results.append({"PAYER": name, "month": month_date, "prob": prob})
        except Exception as e:
            print(f"{name}: не удалось посчитать — {e}")

    if results:
        out = pd.DataFrame(results)
        out["prob_%"] = (out["prob"] * 100).round(1)
        print(out[["PAYER", "month", "prob_%"]].to_string(index=False))
