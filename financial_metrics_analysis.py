from openbb import obb
import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np

ticker = "MSFT"

QUARTERS_NEEDED = {
    "Q2 2024": "2024-06-30",
    "Q1 2024": "2024-03-31",
    "Q4 2023": "2023-12-31",
    "Q3 2023": "2023-09-30",
    "Q2 2023": "2023-06-30",
}

FISCAL_YEARS = {
    "FY 2024": "2024-06-30",
    "FY 2023": "2023-06-30",
    "FY 2022": "2022-06-30",
    "FY 2021": "2021-06-30",
}

def calculate_financial_metrics(ticker):
    print(f" Analyzing {ticker}...")

    try:

        income_q = obb.equity.fundamental.income(
            symbol=ticker,
            peiod="quarter",
            limit=12,
            provider="fmp"
        )

        balance_q = obb.equity.fundamental.balance(
            symbol=ticker,
            period="quarter",
            limit=12,
            provider="fmp"
        )

        cash_q = obb.equity.fundamental.cash(
            symbol=ticker,
            period="quarter",
            limit=12,
            provider="fmp"
        )

        income_a = obb.equity.fundamental.income(
            symbol=ticker,
            period="annual",
            limit=4,
            provider="fmp"
        )

        historical_price = obb.equity.price.historical(
            symbol=ticker,
            start_date="2024-06-28",
            end_date="2024-07-02",
            provider="fmp"
        )

        metrics = obb.equity.fundamental.metrics(
            symbol=ticker,
            period="quarter",
            limit=8,
            provider="fmp"
        )

        if not all([income_q.results, balance_q.results, cash_q.results]):
            raise ValueError("Insufficient data from OpenBB")

        analysis_date = "2024-06-30"

        quarterly_data = {}

        for quarter_name, target_date in QUARTERS_NEEDED.items():
            quarterly_data[quarter_name] = {
                "income": None,
                "balance": None,
                "cash": None,
                "metrics": None
            }

            for inc in income_q.results:
                if hasattr(inc, 'period_ending') and target_date in str(inc.period_ending):
                    quarterly_data[quarter_name]["income"] = inc
                    break

            for bal in balance_q.results:
                if hasattr(bal, 'period_ending') and target_date in str(bal.period_ending):
                    quarterly_data[quarter_name]["balance"] = bal
                    break

            for cf in cash_q.results:
                if hasattr(cf, 'period_ending') and target_date in str(cf.period_ending):
                    quarterly_data[quarter_name]["cash"] = cf
                    break

            for met in metrics.results:
                if hasattr(met, 'period_ending') and target_date in str(met.period_ending):
                    quarterly_data[quarter_name]["metrics"] = met
                    break

            if quarterly_data[quarter_name]["income"]:
                print(f" Found {quarter_name} data")

        annual_data = {}
        for fy_name, target_date in FISCAL_YEARS.items():
            for inc in income_a.results:
                if hasattr(inc, 'period_ending') and target_date in str(inc.period_ending):
                    annual_data[fy_name] = inc
                    break

        print(f" Found {len(annual_data)} years of annual data")

        price_june_2024 = 0
        if historical_price.results:
            for price_data in historical_price.results:
                price_june_2024 = getattr(price_data, 'close', 0) or 0
                if price_june_2024 > 0:
                    break
        #Fallback: use approximate price
        if price_june_2024 == 0: 
            price_june_2024 = 450
            print(f" Using approximate price: ${price_june_2024}")
        else:
            print(f" Price on June 30, 2024: ${price_june_2024}")

        shares_june_2024 = 0
        shares_june_2023 = 0

        if quarterly_data["Q2 2024"]["income"]:
            shares_june_2024 = getattr(quarterly_data["Q2 2024"]["income"], 'weighted_average_basic_shares_outstanding', 0) or 7.43e9
        else:
            shares_june_2024 = 7.43e9 # Approximate shares MSFT

        if quarterly_data["Q2 2023"]["income"]:
            shares_june_2023 = getattr(quarterly_data["Q2 2023"]["income"], 'weighted_average_basic_shares_outstanding', 0) or 7.45e9
        else:
            shares_june_2023 = 7.45e9

        market_cap_june_2024 = price_june_2024 * shares_june_2024

        ttm_quarters = ["Q3 2023", "Q4 2023", "Q1 2024", "Q2 2024"]

        revenue_ttm = 0
        operating_income_ttm = 0
        depreciation_ttm = 0
        operating_cash_flow_ttm = 0
        capex_ttm = 0
        interest_expense_ttm = 0
        tax_expense_ttm = 0
        income_before_tax_ttm = 0
        rd_expense_ttm = 0
        gross_profit_ttm = 0

        for q in ttm_quarters:
            if quarterly_data[q]["income"]:
                inc = quarterly_data[q]["income"]
                revenue_ttm += getattr(inc, 'revenue', 0) or 0
                operating_income_ttm += getattr(inc, 'total_operating_income', 0) or 0
                depreciation_ttm += getattr(inc, 'depreciation_and_amortization', 0) or 0
                interest_expense_ttm += getattr(inc, 'interest_expense', 0) or 0
                tax_expense_ttm += getattr(inc, 'income_tax_expense', 0) or 0
                income_before_tax_ttm += getattr(inc, 'income_before_tax', 0) or 0
                rd_expense_ttm += getattr(inc, 'research_and_development_expense', 0) or 0
                gross_profit_ttm += getattr(inc, 'gross_profit', 0) or 0

            if quarterly_data[q]["cash"]:
                cf = quarterly_data[q]["cash"]
                operating_cash_flow_ttm += getattr(cf, 'operating_cash_flow', 0) or 0
                capex_ttm += abs(getattr(cf, 'capital_expenditure', 0) or 0)

        q2_2024_balance = quarterly_data["Q2 2024"]["balance"]
        q3_2023_balance = quarterly_data["Q3 2023"]["balance"]

        if q2_2024_balance:
            total_assets = getattr(q2_2024_balance, 'total_assets', 0) or 0
            total_liabilities = getattr(q2_2024_balance, 'total_liabilities', 0) or 0
            current_assets = getattr(q2_2024_balance, 'total_current_assets', 0) or 0
            current_liabilities = getattr(q2_2024_balance, 'total_current_liabilities', 0) or 0
            long_term_debt = getattr(q2_2024_balance, 'long_term_debt', 0) or 0
            short_term_debt = getattr(q2_2024_balance, 'short_term_debt', 0) or 0
            cash_equivalents = getattr(q2_2024_balance, 'cash_and_cash_equivalents', 0) or 0
        else:
            total_assets = 0
            total_liabilities = 0
            current_assets = 0
            current_liabilities = 0
            long_term_debt = 0
            short_term_debt = 0
            cash_equivalents = 0

        if q3_2023_balance:
            prior_total_assets = getattr(q3_2023_balance, 'total_assets', 0) or total_assets
            prior_current_liabilities = getattr(q3_2023_balance, 'total_current_liabilities', 0) or current_liabilities
            prior_short_term_debt = getattr(q3_2023_balance, 'short_term_debt', 0) or short_term_debt
        else:
            prior_total_assets = total_assets
            prior_current_liabilities = current_liabilities
            prior_short_term_debt = short_term_debt

        total_debt = long_term_debt + short_term_debt
        enterprise_value = market_cap_june_2024 + total_debt - cash_equivalents
        ebitda_ttm = operating_income_ttm + depreciation_ttm
        ev_ebitda = round(enterprise_value / ebitda_ttm, 2) if ebitda_ttm > 0 else "N/A"
        print(f" EV/EBITDA: {ev_ebitda}")

        tax_rate = tax_expense_ttm / income_before_tax_ttm if income_before_tax_ttm > 0 else 0.21
        nopat = operating_income_ttm * (1 - tax_rate)

        ic_june_2024 = total_assets - current_liabilities + short_term_debt
        ic_sept_2023 = prior_total_assets - prior_current_liabilities + prior_short_term_debt
        avg_invested_capital = (ic_june_2024 + ic_sept_2023) / 2

        roic = round((nopat / avg_invested_capital) * 100, 2) if avg_invested_capital > 0 else "N/A"
        print(f" ROIC: {roic}%")

        free_cash_flow_ttm = operating_cash_flow_ttm - capex_ttm
        fcf_yield = round((free_cash_flow_ttm / market_cap_june_2024) * 100, 2) if market_cap_june_2024 > 0 else "N/A"
        print(f" FCF yield: {fcf_yield}%")

        gross_margin = round((gross_profit_ttm / revenue_ttm) * 100, 2) if revenue_ttm > 0 else "N/A"
        print(f" Gross Margin: {gross_margin}%")

        debt_ebitda = round(total_debt / ebitda_ttm, 2) if ebitda_ttm > 0 else "N/A"
        print(f" Debt/EBITDA: {debt_ebitda}")

        working_capital = round((current_assets - current_liabilities) / 1e9, 2)
        print(f" Working Capital: {working_capital}")

        if "FY 2024" in annual_data and "FY 2021" in annual_data:
            revenue_fy2024 = getattr(annual_data["FY 2024"], 'revenue', 0) or 0
            revenue_fy2021 = getattr(annual_data["FY 2021"], 'revenue', 0) or 0
            if revenue_fy2021 > 0:
                revenue_cagr = round(((revenue_fy2024 / revenue_fy2021) ** (1/3) - 1) * 100, 2)
            else:
                revenue_cagr = "N/A"
            print(f" FY2024 Revenue: ${revenue_fy2024/1e9:.2f}B")
            print(f" FY2021 Revenue: ${revenue_fy2021/1e9:.2f}B")
        else: 
            revenue_cagr = "N/A"
        print(f" 3-Year Revenue CAGR: {revenue_cagr}%")

        rd_intensity = round((rd_expense_ttm / revenue_ttm) * 100, 2) if revenue_ttm > 0 and rd_expense_ttm > 0 else "N/A"
        print(f" R&D Intensity: {rd_intensity}%")

        share_buyback = round(((shares_june_2023 - shares_june_2024) / shares_june_2023) * 100, 2) if shares_june_2023 > 0 else "N/A"
        print(f" Share Buyback: {share_buyback}%")

        if interest_expense_ttm > 0:
            interest_coverage = round(operating_income_ttm / interest_expense_ttm, 2)
        elif interest_expense_ttm == 0 and operating_income_ttm > 0:
            interest_coverage = "No Debt"
        else:
            interest_coverage = "N/A"
        print(f" Interest Coverage: {interest_coverage}")

        book_value = total_assets - total_liabilities
        book_value_per_share = round(book_value / shares_june_2024, 2) if shares_june_2024 > 0 else "N/A"
        print(f" Book Value/Share: ${book_value_per_share}")

        result = {
            "Ticker": ticker,
            "Analysis Date": analysis_date,
            "EV/EBITDA": ev_ebitda,
            "ROIC (%)": roic,
            "FCF Yield (%)": fcf_yield,
            "Gross Margin (%)": gross_margin,
            "Debt/EBITDA": debt_ebitda,
            "Working Capital (B)": working_capital,
            "Revenue CAGR (%)": revenue_cagr,
            "R&D Intensity (%)": rd_intensity,
            "Share Buyback (%)": share_buyback,
            "Interest Coverage": interest_coverage,
            "Book Value/Share": book_value_per_share
        }

        return result

    except Exception as e:
        print(f" Error in Analysis: {e}")
        traceback.print_exc()

        return {
            "Ticker": ticker,
            "Analysis Date": "2024-06-30",
            "EV/EBITDA": "N/A",
            "ROIC (%)": "N/A",
            "FCF Yield (%)": "N/A",
            "Gross Margin (%)": "N/A",
            "Debt/EBITDA": "N/A",
            "Working Capital (B)": "N/A",
            "Revenue CAGR (%)": "N/A",
            "R&D Intensity (%)": "N/A",
            "Share Buyback (%)": "N/A",
            "Interest Coverage": "N/A",
            "Book Value/Share": "N/A"
        }
        
result = calculate_financial_metrics(ticker)

df = pd.DataFrame([result])

print("=" * 70)
print(df.to_string(index=False))

df_to_csv(df)
