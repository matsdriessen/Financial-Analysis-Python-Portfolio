import pandas as pd
from openbb import obb

tickers = ["NVDA","AAPL","XOM","EBAY","AMZN","CSCO","COST","EIX","EA"]
results = []

for ticker in tickers:
    
    balanceall = obb.equity.fundamental.balance(symbol=ticker, limit=100, provider='fmp').results
    incomeall = obb.equity.fundamental.income(symbol=ticker, limit=100, provider='fmp').results
    ratiosall = obb.equity.fundamental.ratios(symbol=ticker, limit=100, provider='fmp').results

    balance = [next(x for x in balanceall if x.fiscal_year == 2024)]
    income = [next(x for x in incomeall if x.fiscal_year == 2024)]
    ratios = [next(x for x in ratiosall if x.fiscal_year == 2024)]

    # Balance sheet items
    ta = balance[0].total_assets
    wc = balance[0].total_current_assets - balance[0].total_current_liabilities
    re = balance[0].retained_earnings
    current_assets = balance[0].total_current_assets
    current_liabilities = balance[0].total_current_liabilities

    # Income statement items
    ebit = income[0].ebitda - income[0].depreciation_and_amortization
    rev = income[0].revenue

    # Ratios
    prcbk = round(ratios[0].price_book_value_ratio, 4)
    eqmlt = round(ratios[0].company_equity_multiplier, 4)

    # Calculate components
    x1 = round(wc/ta, 3)
    x2 = round(re/ta, 3)
    x3 = round(ebit/ta, 3)
    x4 = round(prcbk/(eqmlt-1), 3)
    x5 = round(rev/ta, 3)

    # Calculate Z-score
    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1 * x5

    # Calculate Current Ratio
    current_ratio = current_assets / current_liabilities

    # Categorize
    if z < 1.81:
        Category = "Distress"
    elif 1.81 <= z <= 2.99:
        Category = "Grey Zone"
    else:
        Category = "Safe Zone"

    results.append({
        "Ticker": ticker,
        "Z-score": z,
        "Current Ratio": current_ratio,
        "Category": Category
    })

# Create dataframe and round
dfres = pd.DataFrame(results).round(2)

# Sort by Z-score descending
dfres = dfres.sort_values("Z-score", ascending=False).reset_index(drop=True)

df_to_csv(dfres)
