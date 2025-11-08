import pandas as pd
import numpy as np
from openbb import obb

TICKERS = ["EEM", "IWM", "QQQ", "SPY"]
FETCH_START = "2018-09-01"
FETCH_END = "2019-12-31"

rows = []

for sym in TICKERS:
    # Fetch data
    df = obb.equity.price.historical(
        symbol=sym,
        start_date=FETCH_START,
        end_date=FETCH_END,
        provider="fmp",
        adjustment="splits_and_dividends",
    ).to_dataframe().reset_index()
    
    # Ensure date column exists
    if 'date' not in df.columns:
        df = df.rename(columns={df.columns[0]: 'date'})
    
    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    
    # Step 1: Calculate log returns
    df["logret"] = np.log(df["close"] / df["close"].shift(1))
    
    # Step 2: Calculate rolling volatilities (using ddof=1, which is pandas default)
    df["vol21"] = df["logret"].rolling(window=21, min_periods=21).std()
    df["vol63"] = df["logret"].rolling(window=63, min_periods=63).std()
    
    # Step 3: Filter to 2019 only
    yr = df[(df["date"] >= "2019-01-01") & (df["date"] <= "2019-12-31")].copy()
    
    # Handle case where all volatilities are NaN
    if yr["vol21"].isna().all():
        rows.append({
            "Ticker": sym,
            "Max21dVolPct": 0.0,
            "DateOfMax21d": "N/A",
            "Max63dVolPct": 0.0,
            "DateOfMax63d": "N/A",
            "AboveMedianVolDays": 0,
            "VolRatioAtMax21d": np.nan
        })
        continue
    
    # Step 4: Calculate output metrics
    
    # 1. Max 21-day volatility
    max_vol21 = yr["vol21"].max()
    max_vol21_pct = round(max_vol21 * 100, 4)
    
    # 2. Date of max 21-day volatility (earliest)
    date_of_max21 = yr.loc[yr["vol21"] == max_vol21, "date"].min()
    date_of_max21_str = date_of_max21.strftime("%Y-%m-%d")
    
    # 3. Max 63-day volatility
    if yr["vol63"].isna().all():
        max_vol63_pct = 0.0
        date_of_max63_str = "N/A"
    else:
        max_vol63 = yr["vol63"].max()
        max_vol63_pct = round(max_vol63 * 100, 4)
        date_of_max63 = yr.loc[yr["vol63"] == max_vol63, "date"].min()
        date_of_max63_str = date_of_max63.strftime("%Y-%m-%d")
    
    # 5. Above median days
    median_vol21 = yr["vol21"].median()
    above_median_count = int((yr["vol21"] > median_vol21).sum())
    
    # 6. Volatility ratio at max 21d date
    row_at_max21 = yr[yr["date"] == date_of_max21].iloc[0]
    vol21_at_max = row_at_max21["vol21"]
    vol63_at_max = row_at_max21["vol63"]
    
    if pd.isna(vol63_at_max) or vol63_at_max == 0:
        vol_ratio = np.nan
    else:
        vol_ratio = round(vol21_at_max / vol63_at_max, 4)
    
    rows.append({
        "Ticker": sym,
        "Max21dVolPct": max_vol21_pct,
        "DateOfMax21d": date_of_max21_str,
        "Max63dVolPct": max_vol63_pct,
        "DateOfMax63d": date_of_max63_str,
        "AboveMedianVolDays": above_median_count,
        "VolRatioAtMax21d": vol_ratio
    })

# Create output and sort
out = pd.DataFrame(rows).sort_values("Ticker").reset_index(drop=True)

df_to_csv(out)
