import pandas as pd
import datetime
from pathlib import Path

from normalization import norm_specialty


class ClaimsPreprocessor:

    def __init__(
        self,
        path: Path,
        fill_unknown: bool = True,
    ) -> None:
        self.path = path
        self.fill_unknown = fill_unknown
        self.df: pd.DataFrame | None = None

    def get_df(self) -> pd.DataFrame:
        if self.df is None:
            raise RuntimeError("Call load() first")
        return self.df

    def load(self) -> 'ClaimsPreprocessor':
        self.df = pd.read_csv(self.path)
        return self

    def preprocess(self) -> 'ClaimsPreprocessor':
        if self.df is None:
            raise RuntimeError("Call load() first")

        df = self.df.copy()
        df["MONTH"] = df["MONTH"].apply(self._parse_month)

        bad_mask = df["MONTH"].isna()
        dropped = int(bad_mask.sum())
        if dropped:
            print(f"Dropped rows with invalid MONTH: {dropped}")
        df = df[~bad_mask].reset_index(drop=True)

        df["PAID_AMOUNT"] = pd.to_numeric(df["PAID_AMOUNT"], errors="coerce")

        df["SERVICE_CATEGORY"] = df["SERVICE_CATEGORY"].replace({
            "SpecialistsFFS": "SpecialistFFS"
        })
        df = df[~df['SERVICE_CATEGORY'].isin(['PCPEncounter'])].copy()

        for col in ["SERVICE_CATEGORY", "CLAIM_SPECIALTY", "PAYER"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                if self.fill_unknown:
                    df[col] = df[col].replace({"": "Unknown", "nan": "Unknown"})

        df["CLAIM_SPECIALTY"] = df["CLAIM_SPECIALTY"].apply(norm_specialty)
        self.df = df.reset_index(drop=True)
        return self

    def _parse_month(self, date: datetime.datetime):
        if pd.isna(date):
            return pd.NaT
        try:
            string_date = str(date).strip()
            year, month = int(string_date[:4]), int(string_date[4:6])
            return pd.Timestamp(year=year, month=month, day=1)
        except Exception as exc:
            return pd.NaT
