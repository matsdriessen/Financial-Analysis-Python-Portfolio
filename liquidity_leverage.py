from openbb import obb
import pandas as pd
import os
import numpy as np

YEAR     = 2023
TICKERS  = ["AMD", "MSFT", "NVDA", "HPQ"]     
PROVIDER = "fmp"

def _to_df(obb_object):
    res = getattr(obb_object, "results", None)
    if res is None:
        return pd.DataFrame()
    if isinstance(res, pd.DataFrame):
        return res.copy()
    try:
        rows = []
        seq = res if isinstance(res, (list, tuple)) else [res]
        for r in seq:
            if hasattr(r, "model_dump"):
                rows.append(r.model_dump())
            elif hasattr(r, "dict"):
                rows.append(r.dict())
            else:
                rows.append(r)
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(res)

def _first_col(df, candidates):
    if df is None or df.empty:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        k = c.lower()
        if k in low:
            return low[k]
    for c in df.columns:
        lc = c.lower().replace("-", "_").replace(" ", "_").replace("/", "_")
        for cand in candidates:
            ck = cand.lower().replace("-", "_").replace(" ", "_").replace("/", "_")
            if ck in lc:
                return c
    return None

def _filter_fy(df, year):
    if df is None or df.empty:
        return pd.DataFrame()
    for col in ["fiscal_year","fiscalYear","calendar_year","calendarYear","year"]:
        if col in df.columns:
            try:
                return df[df[col].astype(str) == str(year)]
            except Exception:
                pass
    dcol = _first_col(df, ["date","report_date","reportDate","filing_date","filingDate","period","period_end","periodEndDate"])
    if dcol and dcol in df.columns:
        s = df[dcol].astype(str)
        m = s.str.contains(str(year), na=False)
        if m.any():
            return df[m]
    return pd.DataFrame()

def _row_for_year(df, year):
    d = _filter_fy(df, year)
    return d.iloc[0] if not d.empty else pd.Series(dtype="float64")

def _to_num(x):
    try:
        return float(pd.to_numeric(x))
    except Exception:
        return np.nan

def _sdiv(a, b):
    a = _to_num(a); b = _to_num(b)
    if np.isnan(a) or np.isnan(b) or b == 0.0:
        return np.nan
    return a / b

def compute_one(symbol):
    try:
        bal = _to_df(obb.equity.fundamental.balance(symbol=symbol, provider=PROVIDER, period="annual", limit=10))
        inc = _to_df(obb.equity.fundamental.income (symbol=symbol, provider=PROVIDER, period="annual", limit=10))

        b = _row_for_year(bal, YEAR)
        i = _row_for_year(inc, YEAR)

        # Balance sheet (FY2023)
        ca   = b.get(_first_col(bal, ["total_current_assets","current_assets","currentAssets"]))
        cl   = b.get(_first_col(bal, ["total_current_liabilities","current_liabilities","currentLiabilities"]))
        inv  = b.get(_first_col(bal, ["inventory","inventories","inventory_net","inventoryNet"]))
        cash = b.get(_first_col(bal, [
            "cash_and_cash_equivalents","cashAndCashEquivalents"
        ]))  # primary per spec; fallback below if missing
        if pd.isna(cash):
            cash = b.get(_first_col(bal, [
                "cash_and_short_term_investments","cashAndShortTermInvestments",
                "cashAndCashEquivalentsAndShortTermInvestments"
            ]))

        ta   = b.get(_first_col(bal, ["total_assets","totalAssets"]))
        tl   = b.get(_first_col(bal, ["total_liabilities","totalLiabilities"]))
        te   = b.get(_first_col(bal, [
            "totalStockholdersEquity", "totalEquity"
        ]))
        if pd.isna(te) and (not pd.isna(ta)) and (not pd.isna(tl)):
            te = _to_num(ta) - _to_num(tl)

        total_debt = b.get(_first_col(bal, ["total_debt","totalDebt","shortLongTermDebtTotal"]))
        if pd.isna(total_debt):
            sd = b.get(_first_col(bal, ["short_term_debt","shortTermDebt"]))
            ld = b.get(_first_col(bal, ["long_term_debt","longTermDebt","longTermDebtNoncurrent"]))
            total_debt = _to_num(sd) + _to_num(ld) if (not pd.isna(sd) or not pd.isna(ld)) else np.nan

        # Income statement (FY2023)
        revenue = i.get(_first_col(inc, ["revenue","total_revenue","sales","totalRevenue","Revenue"]))
        gross   = i.get(_first_col(inc, ["gross_profit","grossProfit"]))
        if pd.isna(gross):
            cor = i.get(_first_col(inc, ["cost_of_revenue","costOfRevenue","cost_of_goods_sold"]))
            if not pd.isna(revenue) and not pd.isna(cor):
                gross = _to_num(revenue) - _to_num(cor)
        
        op_income = i.get(_first_col(inc, ["total_operating_income","totalOperatingIncome","ebit"]))
        net_income = i.get(_first_col(inc, ["net_income","netIncome"]))
        interest_expense = i.get(_first_col(inc, ["interest_expense","interestExpense"]))

        # Metrics
        current_ratio = _sdiv(ca, cl)

        quick_ratio = np.nan
        if not pd.isna(ca) and not pd.isna(inv) and not pd.isna(cl):
            quick_ratio = _sdiv(_to_num(ca) - _to_num(inv), cl)

        cash_ratio = _sdiv(cash, cl)
        debt_equity = _sdiv(total_debt, te)

        interest_coverage = _sdiv(op_income, interest_expense)

        asset_turnover = _sdiv(revenue, ta)

        roe_pct = np.nan
        if not pd.isna(net_income) and not pd.isna(te) and _to_num(te) != 0.0:
            roe_pct = ((_to_num(net_income) / _to_num(te)) * 100.0)

        net_margin_pct = np.nan
        if not pd.isna(net_income) and not pd.isna(revenue) and _to_num(revenue) != 0.0:
            net_margin_pct = (_to_num(net_income) / _to_num(revenue)) * 100.0

        gm_pct = np.nan
        if not pd.isna(gross) and not pd.isna(revenue) and _to_num(revenue) != 0.0:
            gm_pct = (_to_num(gross) / _to_num(revenue)) * 100.0

        op_margin_pct = np.nan
        if not pd.isna(op_income) and not pd.isna(revenue) and _to_num(revenue) != 0.0:
            op_margin_pct = (_to_num(op_income) / _to_num(revenue)) * 100.0

        wc_b = np.nan
        if not pd.isna(ca) and not pd.isna(cl):
            wc_b = (_to_num(ca) - _to_num(cl)) / 1e9

        cr_val = _to_num(current_ratio)
        de_val = _to_num(debt_equity)

        if not np.isnan(cr_val) and not np.isnan(de_val):
            if cr_val > 2.0 and de_val < 0.5:
                health = "Strong"
            elif cr_val > 1.5 or de_val < 1.0:
                health = "Moderate"
            else:
                health = "Weak"
        else:
            "N/A"
        
        roe_val = _to_num(roe_pct)
        wc_val = _to_num(wc_b)

        if not np.isnan(roe_val) and not np.isnan(wc_val):
            if roe_val > 15.0 and wc_val > 5.0:
                efficiency = "Efficient"
            elif roe_val > 10.0 or wc_val > 2.0:
                efficiency = "Moderate"
            else:
                efficiency = "Inefficient"
        else:
            efficiency = "N/A"

        def _fmt(x):
            return "N/A" if (x is None or (isinstance(x, float) and np.isnan(x))) else round(float(x), 1)

        return {
            "Ticker": symbol,
            "Current Ratio": _fmt(current_ratio),
            "Quick Ratio": _fmt(quick_ratio),
            "Cash Ratio": _fmt(cash_ratio),
            "Debt/Equity": _fmt(debt_equity),
            "Interest Coverage": _fmt(interest_coverage),
            "Asset Turnover": _fmt(asset_turnover),
            "ROE (%)": _fmt(roe_pct),
            "Net Margin (%)": _fmt(net_margin_pct),
            "Gross Margin (%)": _fmt(gm_pct),
            "Operating Margin (%)": _fmt(op_margin_pct),
            "Working Capital ($B)": _fmt(wc_b),
            "Financial Health": health,
            "Efficiency Category": efficiency,
        }
    except Exception as e:
        print(f"Failed {symbol}: {e}")
        return {
            "Ticker": symbol,
            "Current Ratio": "N/A",
            "Quick Ratio": "N/A",
            "Cash Ratio": "N/A",
            "Debt/Equity": "N/A",
            "Interest Coverage": "N/A",
            "Asset Turnover": "N/A",
            "ROE (%)": "N/A",
            "Net Margin (%)": "N/A",
            "Gross Margin (%)": "N/A",
            "Operating Margin (%)": "N/A",
            "Working Capital ($B)": "N/A",
            "Financial Health": "N/A",
            "Efficiency Category": "N/A",
        }


rows = [compute_one(sym) for sym in TICKERS]

final_df = pd.DataFrame(rows, columns=[
    "Ticker",
    "Current Ratio",
    "Quick Ratio",
    "Cash Ratio",
    "Debt/Equity",
    "Interest Coverage",
    "Asset Turnover",
    "ROE (%)",
    "Net Margin (%)",
    "Gross Margin (%)",
    "Operating Margin (%)",
    "Working Capital ($B)",
    "Financial Health",
    "Efficiency Category",
])

health_order = {'Strong': 0, 'Moderate': 1, 'Weak': 2, 'N/A': 3}
efficiency_order = {'Efficient': 0, 'Moderate': 1, 'Inefficient': 2, 'N/A': 3}

final_df['health_sort'] = final_df['Financial Health'].map(health_order)
final_df['efficiency_sort'] = final_df['Efficiency Category'].map(efficiency_order)

def _sort_val(x):
    if x == "N/A":
        return -999999
    return float(x)

final_df['roe_sort'] = final_df['ROE (%)'].apply(_sort_val)

final_df = final_df.sort_values(
    ['health_sort', 'efficiency_sort', 'roe_sort'], 
    ascending=[True, True, False])

final_df = final_df.drop(['health_sort', 'efficiency_sort', 'roe_sort'], axis=1).reset_index(drop=True)

print(final_df.to_string(index=False))

df_to_csv(final_df)
