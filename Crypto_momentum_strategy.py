from openbb import obb
import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np

CRYPTOS    = ["BTCUSD", "ETHUSD", "SOLUSD"]
START_DATE = "2024-01-01"
END_DATE   = "2024-12-31"
PROVIDER   = "fmp"
SEED_CAP   = 10_000.00

results_list = []

for CRYPTO in CRYPTOS:
    # --- Fetch & prep ---
    hist = obb.crypto.price.historical(
        symbol=CRYPTO,
        start_date=START_DATE,
        end_date=END_DATE,
        provider=PROVIDER
    )

    df = hist.to_df().copy()              # keep same source/shape as original
    n  = len(df)
    position = np.zeros(n, dtype=float)

    # Sum of prior 5 daily percent changes; signal is evaluated at open of day t
    df["sum5"] = df["change_percent"].rolling(window=5, min_periods=5).sum().shift(1)
    df["signal"] = df["sum5"] > 0.03

    # --- Build 5-day non-overlapping long windows when signal is True ---
    last_held_thru = -1
    num_trades = 0
    for i, sig in enumerate(df["signal"].to_numpy()):
        if i <= last_held_thru:
            continue
        if bool(sig):
            start = i
            end   = min(start + 4, n - 1)   # exactly 5 trading days: t..t+4
            position[start:end+1] = 1.0
            last_held_thru = end
            num_trades += 1

    df["position"] = position
    df["return"] = df["position"] * df["change_percent"]

    # Compound through calendar
    cum_growth = (1.0 + df["return"]).cumprod()
    final_value = float(cum_growth.iloc[-1] * SEED_CAP)

    if num_trades > 0:
        total_return_pct = ((final_value - SEED_CAP) / SEED_CAP) * 100
        avg_return_per_trade = total_return_pct / num_trades
    else:
        avg_return_per_trade = 0.0

    if final_value > 12000:
        performance = "Strong Performer"
    elif final_value >= 10000:
        performance = "Market Performer"
    else:
        performance = "Underperformer"

    results_list.append([
        CRYPTO, 
        round(final_value, 2),
        num_trades,
        round(avg_return_per_trade, 2),
        performance
    ])

# --- Required output ---
results = pd.DataFrame(
    results_list,
    columns=["Crypto", "Final Value of the Portfolio", "Number of Trades", "Average Return per Trade (%)", "Performance Category"]
)

results = results.sort_values("Final Value of the Portfolio", ascending=False).reset_index(drop=True)

df_to_csv(results)
