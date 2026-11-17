"""Build data/nse_tri_daily.csv from the raw NSE Indices downloads in data/raw_nse/ (one JSON per index, as returned
by niftyindices.com's 'Total returns Index Values' report, fetched in 1-year chunks).

Columns: date, nifty50, nifty_bank, nifty_it, nifty_fmcg, nifty_pharma (official Total Returns Index levels,
dividends reinvested).  Usage:  cd src && python prep_nse_tri.py
"""
import json
import pandas as pd
from config import DATA

FILES = {"nifty50": "NIFTY_50", "nifty_bank": "NIFTY_BANK", "nifty_it": "NIFTY_IT",
         "nifty_fmcg": "NIFTY_FMCG", "nifty_pharma": "NIFTY_PHARMA"}

cols = {}
for col, stem in FILES.items():
    recs = json.load(open(DATA / "raw_nse" / f"{stem}_tri_raw.json", encoding="utf-8"))
    s = pd.Series({pd.to_datetime(r["Date"], format="%d %b %Y"): float(str(r["TotalReturnsIndex"]).replace(",", ""))
                   for r in recs if str(r.get("TotalReturnsIndex", "")).strip() not in ("", "-")}).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    cols[col] = s
    print(f"{col:13s} {len(s):5d} days  {s.index[0].date()} .. {s.index[-1].date()}")

df = pd.DataFrame(cols)
df.index.name = "date"
df.to_csv(DATA / "nse_tri_daily.csv", float_format="%.2f")
print("rows with all five:", int(df.dropna().shape[0]), "of", len(df))
