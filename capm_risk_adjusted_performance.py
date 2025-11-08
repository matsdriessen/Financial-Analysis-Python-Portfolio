from openbb import obb
import pandas as pd
import numpy as np

tickers = ['GOOGL', 'AAPL', 'NVDA', 'MSFT', 'TSLA', 'META']
market_ticker = '^GSPC'
start_date = '2024-01-01'
end_date = '2024-12-31'
annual_rf_rate = 0.045  # 4.5% annual risk-free rate
daily_rf_rate = annual_rf_rate / 252
trading_days_per_year = 252

obb.user.preferences.output_type = "dataframe"

# Fetch market data
market_data = obb.equity.price.historical(
    symbol=market_ticker, 
    start_date=start_date, 
    end_date=end_date, 
    provider='fmp'
).reset_index()

market_data['date'] = pd.to_datetime(market_data['date']).dt.date
market_data['market_return'] = market_data['adj_close'].pct_change()

results = []

for ticker in tickers:
    # Fetch stock data
    data = obb.equity.price.historical(
        symbol=ticker, 
        start_date=start_date, 
        end_date=end_date, 
        provider='fmp'
    ).reset_index()
    
    data['date'] = pd.to_datetime(data['date']).dt.date
    data['stock_return'] = data['adj_close'].pct_change()
    
    # Merge stock and market returns
    merged_data = pd.merge(
        data[['date', 'stock_return']], 
        market_data[['date', 'market_return']], 
        on='date',
        how='inner'
    )
    
    # Drop rows with NaN values
    merged_data = merged_data.dropna(subset=['stock_return', 'market_return'])
    
    # Calculate beta
    cov = np.cov(merged_data['stock_return'], merged_data['market_return'])[0][1]
    var = np.var(merged_data['market_return'], ddof=0)
    beta = cov / var
    
    # Calculate correlation and R-squared
    correlation = np.corrcoef(merged_data['stock_return'], merged_data['market_return'])[0][1]
    r_squared = correlation ** 2
    
    # Calculate average returns
    avg_stock_return = merged_data['stock_return'].mean()
    avg_market_return = merged_data['market_return'].mean()
    
    # Calculate alpha (Jensen's alpha) - daily then annualize
    daily_alpha = avg_stock_return - (daily_rf_rate + beta * (avg_market_return - daily_rf_rate))
    annual_alpha = daily_alpha * trading_days_per_year * 100  # Convert to percentage
    
    # Calculate Sharpe ratio - annualized
    stock_std = merged_data['stock_return'].std(ddof=0)
    daily_sharpe = (avg_stock_return - daily_rf_rate) / stock_std
    annual_sharpe = daily_sharpe * np.sqrt(trading_days_per_year)
    
    # Calculate Treynor ratio - annualized
    daily_treynor = (avg_stock_return - daily_rf_rate) / beta
    annual_treynor = daily_treynor * trading_days_per_year
    
    # Round values
    beta_rounded = round(beta, 1)
    alpha_rounded = round(annual_alpha, 2)
    sharpe_rounded = round(annual_sharpe, 2)
    treynor_rounded = round(annual_treynor, 2)
    correlation_rounded = round(correlation, 3)
    r_squared_rounded = round(r_squared, 3)
    
    # Categorize by risk (based on beta)
    if beta_rounded > 1.2:
        risk_category = "High Volatility"
    elif beta_rounded >= 0.8:
        risk_category = "Market Volatility"
    else:
        risk_category = "Low Volatility"
    
    # Categorize by performance (based on alpha)
    if alpha_rounded > 2:
        performance_category = "Outperform"
    elif alpha_rounded >= -2:
        performance_category = "Market Perform"
    else:
        performance_category = "Underperform"
    
    results.append({
        'ticker': ticker,
        'beta': beta_rounded,
        'alpha': alpha_rounded,
        'sharpe_ratio': sharpe_rounded,
        'treynor_ratio': treynor_rounded,
        'correlation': correlation_rounded,
        'r_squared': r_squared_rounded,
        'risk_category': risk_category,
        'performance_category': performance_category
    })

# Create output dataframe
final_df = pd.DataFrame(results)

# Define custom sort order for performance_category
category_order = {'Outperform': 0, 'Market Perform': 1, 'Underperform': 2}
final_df['sort_key'] = final_df['performance_category'].map(category_order)

# Sort by performance category, then alpha descending, then ticker ascending
final_df = final_df.sort_values(
    ['sort_key', 'alpha', 'ticker'], 
    ascending=[True, False, True]
).drop('sort_key', axis=1).reset_index(drop=True)

df_to_csv(final_df)
