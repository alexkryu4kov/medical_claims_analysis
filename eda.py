from preprocessor import ClaimsPreprocessor
from pathlib import Path
import numpy as np
import pandas as pd


preprocessor = ClaimsPreprocessor(Path("claims_sample_data.csv")).load().preprocess()
df = preprocessor.get_df()

print(f"Shape: {df.shape[0]} rows × {df.shape[1]} cols")
print("Dtypes:\n", df.dtypes.to_string(), "\n")

print(f"Duplicate full rows: {df.duplicated().sum()}")


amount = df["PAID_AMOUNT"]
pos = float(amount[amount > 0].sum())
neg = float(-amount[amount < 0].sum())
net = float(amount.sum())
print(f"Amounts: pos={pos:,.2f} | neg={neg:,.2f} | net={net:,.2f} | neg_share={(neg/pos) if pos else np.nan:.2%}")
print(f"Rows: pos={(amount>0).sum()} | neg={(amount<0).sum()} | zero={(amount==0).sum()} | NaN={amount.isna().sum()}\n")

# ----- Разрез по SERVICE_CATEGORY -----
col = "SERVICE_CATEGORY"

print(df[col].unique())

# по строкам
rows_total = df.groupby(col).size().rename("rows_total")
rows_zero  = amount.eq(0).groupby(df[col]).sum().rename("rows_zero")
rows_neg   = amount.lt(0).groupby(df[col]).sum().rename("rows_neg")

# по суммам
pos_amt = amount.where(amount > 0, 0.0).groupby(df[col]).sum().rename("pos_amt")
neg_amt = (-amount.where(amount < 0, 0.0)).groupby(df[col]).sum().rename("neg_amt")  # модуль

# сборка
res = pd.concat([rows_total, rows_zero, rows_neg, pos_amt, neg_amt], axis=1).fillna(0)

# доли и итог
res["zero_rows_share"] = res["rows_zero"] / res["rows_total"].replace(0, np.nan)
res["neg_rows_share"]  = res["rows_neg"]  / res["rows_total"].replace(0, np.nan)
res["neg_share_amt"]   = res["neg_amt"]   / res["pos_amt"].replace(0, np.nan)
res["net_amt"]         = res["pos_amt"]   - res["neg_amt"]

# удобный порядок колонок и сортировка
cols_to_show = [
    "rows_total","rows_zero","rows_neg",
    "zero_rows_share","neg_rows_share",
    "pos_amt","neg_amt","neg_share_amt","net_amt"
]
res_sorted = res.sort_values(["neg_share_amt","neg_rows_share"], ascending=False)

print(res_sorted[cols_to_show].to_string())

by_cat = df.groupby("SERVICE_CATEGORY")["PAID_AMOUNT"].sum().sort_values(ascending=False)
monthly_cat = (df.groupby(["MONTH","SERVICE_CATEGORY"])["PAID_AMOUNT"].sum()
                 .unstack(fill_value=0).sort_index())

print(by_cat)
print(monthly_cat)