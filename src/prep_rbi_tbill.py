"""Convert RBI 'Auctions of 91-Day Government of India Treasury Bills' (.xlsx, source: Reserve Bank of India) to date,yield_pct.
date = auction date; yield_pct = implicit yield at cut-off price (percent, actual/365). Rejected auctions ('-') dropped.
Usage:  cd src && python prep_rbi_tbill.py ../data/rbi_91d_tbill_auctions.xlsx ../data/tbill_91d_auctions.csv"""
import sys, datetime as dt
import pandas as pd
from read_xlsx import read
rows = list(read(sys.argv[1]).values())[0]
hdr = rows[2]; i_date, i_y = hdr.index("Date of Auction"), hdr.index("Implicit Yield at Cut-off Price (percent)")
recs, skipped = [], 0
for r in rows[3:]:
    try:
        d = dt.date(1899, 12, 30) + dt.timedelta(days=int(float(r[i_date])))
    except (TypeError, ValueError, IndexError):
        continue                                           # header / fiscal-year label / note rows
    try:
        y = float(r[i_y])
    except (TypeError, ValueError):
        skipped += 1; continue                             # auction with no accepted bids
    recs.append((pd.Timestamp(d), y))
df = pd.DataFrame(recs, columns=["date", "yield_pct"]).sort_values("date")
bad = (df.date < "1990-01-01").sum(); df = df[df.date >= "1990-01-01"]          # column-number row ("1".."16") parses as 1899-12-31
print("dropped non-date rows:", bad)
dups = df.date.duplicated().sum(); df = df.drop_duplicates("date", keep="last")
df.to_csv(sys.argv[2], index=False, date_format="%Y-%m-%d")
print(f"{len(df)} auctions {df.date.min().date()} .. {df.date.max().date()}; skipped (no cut-off) {skipped}; duplicate dates {dups}")
g = df.date.diff().dt.days; print("gap days: median", g.median(), "max", g.max(), "at", df.date[g.idxmax()].date() if g.notna().any() else None)
print("yield range", df.yield_pct.min(), df.yield_pct.max())
s = df.set_index("date").yield_pct
print("annual means 2007-2026:"); print(s["2007":].resample("YE").agg(["mean","min","max","count"]).round(2).to_string())
