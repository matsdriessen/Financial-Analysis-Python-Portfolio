import pandas as pd 
import numpy as np
import traceback
from scipy import stats
from openbb import obb
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

print("Starting financial distress analysis")
print("=" * 50)

ticker = "BA"

QUARTERS = {
    "Q2 2024": "2024-06-30",
    "Q1 2024": "2024-03-31",
    "Q4 2023": "2023-12-31",
    "Q3 2023": "2023-09-30",
    "Q2 2023": "2023-06-30",
    "Q1 2023": "2023-03-31",
    "Q4 2022": "2022-12-31",
    "Q3 2022": "2022-09-30"
}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calculate_distress_score(ticker):
    print(f" Analyzing {ticker}")

    try:

        income_data = obb.equity.fundamental.income(
            symbol=ticker,
            period="quarter",
            limit=20,
            provider="fmp"
        )

        balance_data = obb.equity.fundamental.balance(
            symbol=ticker,
            period="quarter",
            limit=20,
            provider="fmp"
        )

        cash_data = obb.equity.fundamental.cash(
            symbol=ticker,
            period="quarter",
            limit=20,
            provider="fmp"
        )

        if not all([income_data.results, balance_data.results, cash_data.results]):
            raise ValueError(f"Insufficient data for {ticker}")

        revenues = []
        net_incomes = []
        operating_incomes = []
        ebit_values = []
        total_assets = []
        current_assets = []
        current_liabilities = []
        retained_earnings = []
        working_capital = []
        total_liabilities = []
        book_equity = []
        operating_cash_flows = []

        quarters_found = 0

        for i in range(min(8, len(income_data.results))):
            if quarters_found >= 8:
                break

            inc = income_data.results[i]

            bal = None
            if i < len(balance_data.results):
                bal = balance_data.results[i]

            cf = None
            if i < len(cash_data.results):
                cf = cash_data.results
            
            if inc and bal:
                revenues.append(getattr(inc, 'revenue', 0) or 0)
                net_incomes.append(getattr(inc, 'net_income', 0) or 0)
                operating_incomes.append(getattr(inc, 'total_operating_income', 0) or 0)
                ebit_values.append(getattr(inc, 'ebit', 0) or getattr(inc, 'total_operating_income', 0) or 0)

                ta = getattr(bal, 'total_assets', 1) or 1
                ca = getattr(bal, 'total_current_assets', 0) or 0
                cl = getattr(bal, 'total_current_liabilities', 0) or 0
                re = getattr(bal, 'retained_earnings', 0) or 0
                tl = getattr(bal, 'total_liabilities', 1) or 1

                total_assets.append(ta)
                current_assets.append(ca)
                current_liabilities.append(cl)
                retained_earnings.append(re)
                working_capital.append(ca - cl)
                total_liabilities.append(tl)
                book_equity.append(ta - tl)

                if cf:
                    ocf = getattr(cf, 'operating_cash_flow', 0) or 0
                    operating_cash_flows.append(ocf)
                else:
                    operating_cash_flows.append(0)

                quarters_found += 1

                if hasattr(inc, 'period_ending'):
                    print(f" Found Q{quarters_found}: {inc.period_ending}")

        revenues = list(reversed(revenues))
        net_incomes = list(reversed(net_incomes))
        operating_incomes = list(reversed(operating_incomes))
        ebit_values = list(reversed(ebit_values))
        total_assets = list(reversed(total_assets))
        current_assets = list(reversed(current_assets))
        current_liabilities = list(reversed(current_liabilities))
        retained_earnings = list(reversed(retained_earnings))
        working_capital = list(reversed(working_capital))
        total_liabilities = list(reversed(total_liabilities))
        book_equity = list(reversed(book_equity))
        operating_cash_flows = list(reversed(operating_cash_flows))

        print(f" Found {quarters_found}")

        if len(revenues) == 0 or len(total_assets) == 0:
            print(f" Critical: No financial data found!")
            return {
                "Ticker": ticker,
                "Assessment Date": "2024-06-30",
                "Distress Score": 50.0
            }

        z_scores = []
        quarters_to_analyze = min(4, len(total_assets))

        for i in range(-quarters_to_analyze, 0):
            if abs(i) <= len(total_assets) and total_assets[i] != 0:
                z = (
                    0.717 * (working_capital[i] / total_assets[i]) +
                    0.847 * (retained_earnings[i] / total_assets[i]) +
                    3.107 * (ebit_values[i] / total_assets[i]) +
                    0.420 * (book_equity[i] / total_liabilities[i] if total_liabilities[i] != 0 else 1) +
                    0.998 * (revenues[i] / total_assets[i])
                )
            else:
                z = 1.0
            z_normalized = sigmpoid(0.5 * (z - 1.8))
            z_scores.append(z_normalized)

        if len(z_scores) == 4:
            z_weights = [0.1, 0.2, 0.3, 0.4]
        elif len(z_scores) == 3:
            z_weights = [0.2, 0.3, 0.5]
        elif len(z_scores) == 2:
            z_weights = [0.4, 0.6]
        else:
            z_weights = [1.0]

        z_contribution = sum(z x w for z, w in zip(z_scores, z_weights)) * 40
        print(f" Z-Score contribution: {z_contribution:.2f}/40")

        f_score = 0
        data_points = min(len(net_incomes), len(total_assets))

        if data_points > 0:
            if len(net_incomes) > 0 and net_incomes[-1] > 0:
                f_score += 1

            if len(operating_cash_flows) > 0 and operating_cash_flows[-1] > 0:
                f_score += 1

            if data_points >= 2:
                roa_current = net_incomes[-1] / total_assets[-1] if total_assets[-1] != 0 else 0
                roa_prior = net_incomes[-2] / total_assets[-2] if total_assets[-2] != 0 else 0
                f_score += 1 if roa_current > roa_prior else 0

            if len(operating_cash_flows) > 0 and len(net_incomes) > 0:
                f_score += 1 if operating_cash_flows[-1] > net_incomes[-1] else 0

            if data_points >= 2:
                leverage_now = total_liabilities[-1] / total_assets[-1] if total_assets[-1] != 0 else 1
                leverage_before = total_liabilities[-2] / total_assets[-2] if total_assets[-2] != 0 else 1
                f_score += 1 if leverage_now < leverage_before else 0

            if data_points >= 2:
                cr_now = current_assets[-1] / current_liabilities[-1] if current_liabilities[-1] != 0 else 1
                cr_before = current_assets[-2] / current_liabilities[-2] if current_liabilities[-2] != 0 else 1
                f_score += 1 if cr_now > cr_before else 0
            
            f_score += 1 # Assume now new equity

            
 





        for q_name, q_date in QUARTERS.items():
            quarterly_data[q_name] = {
                "income": None,
                "balance": None,
                "cash": None
            }

            for inc in income_data.results:
                if hasattr(inc, 'period_ending') and q_date in str(inc.period_ending):
                    quarterly_data[q_name]["income"] = inc
                    break
            
            for bal in balance_data.results:
                if hasattr(bal, 'period_ending') and q_date in str(bal.period_ending):
                    quarterly_data[q_name]["income"] = bal
                    break

            for cf in cash_data.results:
                if hasattr(cf, 'period_ending') and q_date in str(cf.period_ending):
                    quarterly_data[q_name]["cash"] = cf
                    break

        revenues = []
        net_incomes = []
        operating_incomes = []
        ebit_values = []
        total_assets = []
        current_assets = []
        current_liabilities = []
        retained_earnings = []
        working_capital = []
        total_liabilities = []
        book_equity = []
        operating_cash_flows = []

        quarter_order = ["Q3 2022", "Q4 2022", "Q1 2023", "Q2 2023", "Q3 2023", "Q4 2023", "Q1 2024", "Q2 2024"]

        for q in quarter_order:
            if q in quarterly_data:
                q_data = quarterly_data[q]

                if q_data["income"]:
                    inc = q_data["income"]
                    revenues.append(getattr(inc, 'revenue', 0) or 0)
                    net_incomes.append(getattr(inc, 'consolidated_net_income', 0) or 0)
                    operating_incomes.append(getattr(inc, 'total_operating_income', 0) or 0)
                    ebit_values.append(getattr(inc, 'total_operating_income,', 0) or 0)

                if q_data["balance"]:
                    bal = q_data["balance"]
                    ta = getattr(bal, 'total_assets', 1) or 1
                    ca = getattr(bal, 'total_current_assets', 0) or 0
                    cl = getattr(bal, 'total_current_liabilities', 0) or 0
                    re = getattr(bal, 'retained_earnings', 0) or 0
                    tl = getattr(bal, 'total_liabilities', 1) or 1

                    total_assets.append(ta)
                    current_assets.append(ca)
                    current_liabilities.append(cl)
                    retained_earnings.append(re)
                    working_capital.append(ca - cl)
                    total_liabilities.append(tl)
                    book_equity.append(ta - tl)
                
                if q_data["cash"]:
                    cf = q_data["cash"]
                    ocf = getattr(cf, 'operating_cash_flow', 0) or 0
                    operating_cash_flows.append(ocf)

        print(f" Found data for {len(revenues)} quarters")

        z_scores = []
        for i in range(-4, 0):
            if i < -len(total_assets) or total_assets[i] == 0:
                z = 1.0
            else:
                z = (
                    0.717 * (working_capital[i] / total_assets[i]) +
                    0.847 * (retained_earnings[i] / total_assets[i]) +
                    3.107 * (ebit_values[i] / total_assets[i]) +
                    0.420 * (book_equity[i] / total_liabilities[i] if total_liabilities[i] != 0 else 1) +
                    0.998 * (revenues[i] / total_assets[i])
                )
            z_normalized = sigmoid(0.5 * (z - 1.8))
            z_scores.append(z_normalized)

        z_weights = [0.1, 0.2, 0.3, 0.4]
        z_contribution = sum(z * w for z, w in zip(z_scores, z_weights)) * 40
        print(f" Z-score contribution: {z_contribution:.2f}/40")

        f_score = 0
        if len(net_incomes) >= 8:
            f_score += 1 if net_incomes[-1] > 0 else 0
            f_score += 1 if operating_cash_flows[-1] > 0 else 0

            if len(net_incomes) >= 5 and len(total_assets) >= 5:
                roa_current = net_incomes[-1] / total_assets[-1] if total_assets[-1] != 0 else 0
                roa_prior = net_incomes[-5] / total_assets[-5] if total_assets[-5] != 0 else 0
                f_score += 1 if roa_current > roa_prior else 0

            f_score += 1 if operating_cash_flows[-1] > net_incomes[-1] else 0

            if len(total_liabilities) >= 5:
                leverage_now = total_liabilities[-1] / total_assets[-1] if total_assets[-1] != 0 else 1
                leverage_before = total_liabilities[-5] / total_assets[-5] if total_assets[-5] != 0 else 1
                f_score += 1 if leverage_now < leverage_before else 0

            if len(current_assets) >= 5 and len(current_liabilities) >= 5:
                cr_now = current_assets[-1] / current_liabilities[-1] if current_liabilities[-1] != 0 else 1
                cr_before = current_assets[-5] / current_liabilities[-5] if current_liabilities[-5] != 0 else 1
                f_score += 1 if cr_now > cr_before else 0

            f_score += 1 # Assuming no new equity is issued

            if len(revenues) >= 5:
                margin_now = operating_incomes[-1] / revenues[-1] if revenues[-1] != 0 else 0
                margin_before = operating_incomes[-5] / revenues[-5] if revenues[-5] != 0 else 0
                f_score += 1 if margin_now > margin_before else 0

                turnover_now = revenues[-1] / total_assets[-1] if total_assets[-1] != 0 else 0
                turnover_before = revenues[-5] / total_assets[-5] if total_assets[-5] != 0 else 0
                f_score += 1 if turnover_now > turnover_before else 0

        f_momentum = 0
        if len(net_incomes) >= 4:
            recent_trend = (net_incomes[-1] - net_incomes[-4]) / abs(net_incomes[-4]) if net_incomes[-4] != 0 else 0
            f_momentum = min(3, max(-3, recent_trend))

        adjusted_f_score = f_score + (f_momentum * 0.5)
        f_normalized = adjusted_f_score / 12
        f_contribution = f_normalized * 35
        print(f" F-Score contribution: {f_contribution:.2f}/35 (base: {f_score}/9)")

        m_score = -5
        if len(revenues) >= 5 and reveneues[-5] != 0:
            dsri = (revenues[-1] / revenues[-5])

            gmi = 1
            if operating_incomes[-5] != 0 and revenues[-5] != 0:
                margin_old = operating_incomes[-5] / revenues[-5]
                margin_new = operating_incomes[-1] / revenues[-1] if revenues[-1] != 0 else 0
                gmi = margin_old / margin_new if margin_new != 0 else 1

            aqi = 1
            if len(total_assets) >= 5 and len(current_assets) >= 5:
                non_current_old = (total_assets[-5] - current_assets[-5]) / total_assets[-5] if total_assets[-5] != 0 else 0
                non_current_new = (total_assets[-1] / current_assets[-1]) / total_assets[-1] if total_assets[-1] != 0 else 0
                aqi = non_current_new / non_current_old if non_current_old != 0 else 1

            sgi = revenues[-1] / revenues[-5]

            depi = 1
            if len(operating_incomes) >= 5:
                if revenues[-5] != 0 and revenues[-1] != 0:
                    depr_rate_old = 1 - (operating_incomes[-5] / revenues[-5])
                    depr_rate_new = 1 - (operating_incomes[-1] / revenues[-1])
                    depi = depr_rate_old / depr_rate_new if depr_rate_new != 0 else 1

            sgai = 1
            if revenues[-5] != 0 and revenues[-1] != 0:
                sga_old = (revenues[-5] - operating_incomes[-5]) / revenues[-5]
                sga_new = (revenues[-1] - operating_incomes[-1]) / revenues[-1]
                sgai = sga_new / sga_old if sga_old != 0 else 1

            lvgi = 1
            if total_liabilities[-5] != 0 and total_assets[-5] != 0:
                lev_old = total_liabilities[-5] / total_assets[-5]
                lev_new = total_liabilities[-1] / total_assets[-1] if total_assets[-1] != 0 else 0
                lvgi = lev_new / lev_old if lev_old != 0 else 1 

            tata = 0
            if len(operating_cash_flows) > 0 and total_assets[-1] != 0:
                tata = (net_incomes[-1] - operating_cash_flows[-1]) / total_assets[-1]

            m_score = (-4.84 + 
                0.92 * dsri + 
                0.528 * gmi +
                0.404 * aqi + 
                0.892 * sgi +
                0.115 * depi -
                0.172 * sgai + 
                4.679 * tata - 
                0.327 * lvgi)

        manipulation_prob = sigmoid(m_score)
        quality_score = 1 - manipulation_prob
        m_contribution = quality_score * 25
        print(f" M-Score contribtution: {m_contribution:.2f}/25")

        volatility_penatly = 1.0
        if len(revenues) >= 8:
            rev_vol = np.std(revenues) / np.mean(revenues) if np.mean(revenues) != 0 else 0
            earnings_vol = np.std(net_incomes) / abs(np.mean(net_incomes)) if np.mean(net_incomes) != 0 else 0
            cf_vol = np.std(operating_cash_flows) / abs(np.mean(operating_cash_flows)) if np.mean(operating_cash_flows) != 0 else 0

            total_volatility = rev_vol + earnings_vol + cf_vol
            volatility_penatly = np.exp(-total_volatility)

        print(f" Volatility penalty factor: {volatility_penatly:.3f}")

        base_score = z_contribution + f_contribution + m_contribution

        interaction_multiplier = 1.0
        if z_scores[-1] < sigmoid(0.5 * (1.8 - 1.8)) and f_score < 5:
            interaction_multiplier = 0.7
        elif z_scores[-1] > sigmoid(0.5 * (3 - 1.8)) and f_score > 7:
            interaction_multiplier = 1.2

        adjusted_score = base_score * interaction_multiplier
        volatility_adjusted = adjusted_score * (0.6 + 0.4 * volatility_penatly)

        raw_score = volatility_adjusted
        distress_score = 100 / (1 + np.exp(-0.1 * (raw_score - 50)))
        distress_score = round(distress_score, 1)

        assessment_date = "2024-06-30"

        result = {
            "Ticker": ticker,
            "Assessment Date": assessment_date,
            "Distress Score": distress_score
        }

        print(f" Analysis Complete for {ticker}")
        return result

    except Exception as e:
        print(f" Error in analysis for {ticker}: {e}")
        traceback.print_exc()

        return {
           "Ticker": ticker,
           "Assessment Date": "2024-06-30",
           "Distress Score": 50.0
        }

result = calculate_distress_score(ticker)

final_df = pd.DataFrame([result])

print(final_df.to_string(index=False))

df_to_csv(final_df)

print(" Analysis file saved successfully")
