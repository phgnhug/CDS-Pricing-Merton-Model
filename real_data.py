"""
Real Data Pipeline: Fetch actual market data and run CDS pricing

Data sources:
- Stock prices & volatility: yfinance (free)
- Company debt: User input or CSV
- CDS spreads: User input or CSV (paid sources: Bloomberg, Refinitiv)

Usage:
    python3 real_data.py --tickers AAPL MSFT TSLA --debt-file debt.csv
"""

import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from calibration import calibrate_merton
from merton_model import MertonModel
from cds_pricing import cds_fair_spread_bps
from sensitivity import CDSSensitivity


# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

def fetch_stock_data(ticker, period='1y'):
    """
    Fetch stock price and compute volatility from yfinance
    
    Args:
        ticker (str): Stock ticker (e.g., 'AAPL')
        period (str): Data period (default: 1 year)
    
    Returns:
        dict: {
            'ticker': str,
            'stock_price': float,
            'equity_vol': float,
            'company_name': str
        }
    """
    print(f"  Fetching data for {ticker}...", end=" ")
    
    try:
        # Download historical data
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if len(hist) < 30:
            print("✗ (insufficient data)")
            return None
        
        # Current stock price
        current_price = hist['Close'].iloc[-1]
        
        # Compute volatility from returns
        returns = hist['Close'].pct_change().dropna()
        daily_vol = returns.std()
        annualized_vol = daily_vol * np.sqrt(252)  # 252 trading days per year
        
        # Get company name
        company_name = stock.info.get('longName', ticker)
        
        print(f"✓")
        
        return {
            'ticker': ticker,
            'company_name': company_name,
            'stock_price': current_price,
            'equity_vol': annualized_vol,
            'data_date': datetime.now().strftime('%Y-%m-%d'),
        }
    
    except Exception as e:
        print(f"✗ ({e})")
        return None


def load_debt_from_csv(filename):
    """
    Load company debt from CSV file
    
    CSV format:
        ticker, total_debt
        AAPL, 50
        MSFT, 40
    
    Args:
        filename (str): Path to CSV file
    
    Returns:
        dict: {ticker: debt}
    """
    try:
        df = pd.read_csv(filename)
        debt_dict = dict(zip(df['ticker'], df['total_debt']))
        print(f"✓ Loaded debt data for {len(debt_dict)} companies")
        return debt_dict
    except Exception as e:
        print(f"✗ Failed to load debt data: {e}")
        return {}


def load_cds_from_csv(filename):
    """
    Load CDS spreads from CSV file
    
    CSV format:
        ticker, cds_spread_bps
        AAPL, 45
        MSFT, 35
    
    Args:
        filename (str): Path to CSV file
    
    Returns:
        dict: {ticker: cds_bps}
    """
    try:
        df = pd.read_csv(filename)
        cds_dict = dict(zip(df['ticker'], df['cds_spread_bps']))
        print(f"✓ Loaded CDS data for {len(cds_dict)} companies")
        return cds_dict
    except Exception as e:
        print(f"✗ Failed to load CDS data: {e}")
        return {}


def get_manual_debt_input():
    """
    Interactively get company debt from user
    """
    debt_dict = {}
    
    print("\n" + "="*70)
    print("Enter company debt values (or press Enter to skip)")
    print("="*70)
    
    while True:
        ticker = input("\nTicker (or 'done' to finish): ").upper().strip()
        
        if ticker == 'DONE':
            break
        
        if not ticker:
            continue
        
        try:
            debt = float(input(f"Total debt for {ticker} (billions USD): "))
            debt_dict[ticker] = debt
            print(f"  ✓ {ticker}: ${debt}B")
        except ValueError:
            print("  ✗ Invalid input, please enter a number")
    
    return debt_dict


def get_manual_cds_input():
    """
    Interactively get CDS spreads from user
    """
    cds_dict = {}
    
    print("\n" + "="*70)
    print("Enter market CDS spreads (or press Enter to skip)")
    print("="*70)
    
    while True:
        ticker = input("\nTicker (or 'done' to finish): ").upper().strip()
        
        if ticker == 'DONE':
            break
        
        if not ticker:
            continue
        
        try:
            cds = float(input(f"Market CDS spread for {ticker} (bps): "))
            cds_dict[ticker] = cds
            print(f"  ✓ {ticker}: {cds:.1f} bps")
        except ValueError:
            print("  ✗ Invalid input, please enter a number")
    
    return cds_dict


# ============================================================================
# MAIN PIPELINE FOR REAL DATA
# ============================================================================

def process_real_company(ticker, stock_data, debt, market_cds, 
                         T=1.0, r=0.05, recovery=0.40):
    """
    Process one company with real data
    """
    if stock_data is None:
        return None
    
    company_name = stock_data['company_name']
    S = stock_data['stock_price']
    equity_vol = stock_data['equity_vol']
    
    print(f"\n{'='*70}")
    print(f"{company_name} ({ticker})")
    print(f"{'='*70}")
    
    print(f"\n[1] Market Data (from yfinance)")
    print(f"    Stock Price: ${S:.2f}")
    print(f"    Equity Volatility: {equity_vol:.2%}")
    print(f"    Total Debt: ${debt:.2f}B")
    
    if market_cds is None:
        print(f"    Market CDS: Not available")
        market_cds = np.nan
    else:
        print(f"    Market CDS: {market_cds:.1f} bps")
    
    # Calibration
    print(f"\n[2] Calibration")
    try:
        asset_vol = calibrate_merton(S, equity_vol, debt, T, r)
        print(f"    ✓ Asset Volatility: {asset_vol:.2%}")
    except Exception as e:
        print(f"    ✗ Calibration failed: {e}")
        return None
    
    # Merton Model
    print(f"\n[3] Default Probability")
    model = MertonModel(S, debt, asset_vol, T, r, recovery)
    pd = model.default_probability()
    dd = model.distance_to_default()
    
    print(f"    Distance to Default: {dd:.4f}")
    print(f"    Default Probability: {pd:.4f} ({pd:.2%})")
    
    # CDS Pricing
    print(f"\n[4] CDS Pricing")
    cds_model = cds_fair_spread_bps(pd, recovery, T, r)
    print(f"    Model CDS Spread: {cds_model:.1f} bps")
    
    if not np.isnan(market_cds):
        diff = cds_model - market_cds
        print(f"    Market CDS Spread: {market_cds:.1f} bps")
        print(f"    Difference: {diff:+.1f} bps")
        
        if diff > 0:
            print(f"    → Model overprices (SELL CDS opportunity)")
        else:
            print(f"    → Model underprices (BUY CDS opportunity)")
    
    # Greeks
    print(f"\n[5] Sensitivity (Greeks)")
    sensitivity = CDSSensitivity(S, debt, asset_vol, T, r, recovery)
    vega = sensitivity.delta_volatility()
    delta_r = sensitivity.delta_recovery()
    
    print(f"    Vega: {vega:.2f} bps per 1% vol change")
    print(f"    Delta Recovery: {delta_r:.2f} bps per 1% recovery change")
    
    # Results
    results = {
        'Ticker': ticker,
        'Company': company_name,
        'Stock_Price': S,
        'Equity_Vol': equity_vol,
        'Total_Debt': debt,
        'Asset_Vol': asset_vol,
        'Default_Prob': pd,
        'Distance_to_Default': dd,
        'Model_CDS_bps': cds_model,
        'Market_CDS_bps': market_cds,
        'Vega': vega,
        'Delta_Recovery': delta_r,
        'Data_Date': stock_data['data_date'],
    }
    
    print(f"\n✓ {company_name} processed successfully")
    
    return results


def main():
    """
    Main pipeline for real data
    """
    parser = argparse.ArgumentParser(
        description='CDS Pricing with Real Market Data'
    )
    parser.add_argument(
        '--tickers',
        type=str,
        nargs='+',
        help='Stock tickers (e.g., AAPL MSFT TSLA)',
        default=['AAPL', 'MSFT', 'TSLA']
    )
    parser.add_argument(
        '--debt-file',
        type=str,
        help='CSV file with company debt data'
    )
    parser.add_argument(
        '--cds-file',
        type=str,
        help='CSV file with market CDS spreads'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactively input debt and CDS data'
    )
    parser.add_argument(
        '--maturity',
        type=float,
        default=1.0,
        help='CDS maturity in years (default: 1.0)'
    )
    parser.add_argument(
        '--recovery',
        type=float,
        default=0.40,
        help='Recovery rate (default: 0.40)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/real_data_results.csv',
        help='Output CSV file'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("CDS PRICING WITH REAL MARKET DATA")
    print("="*70)
    print(f"Tickers: {', '.join(args.tickers)}")
    print(f"Maturity: {args.maturity} years")
    print(f"Recovery Rate: {args.recovery:.0%}")
    
    # Fetch stock data
    print(f"\n[Step 1] Fetching stock data from yfinance...")
    stock_data_dict = {}
    
    for ticker in args.tickers:
        data = fetch_stock_data(ticker)
        if data:
            stock_data_dict[ticker] = data
    
    if not stock_data_dict:
        print("✗ No stock data available")
        return
    
    print(f"✓ Fetched data for {len(stock_data_dict)} companies")
    
    # Load or input debt data
    print(f"\n[Step 2] Loading company debt...")
    
    if args.debt_file:
        debt_dict = load_debt_from_csv(args.debt_file)
    elif args.interactive:
        debt_dict = get_manual_debt_input()
    else:
        print("Note: No debt data provided. Use --debt-file or --interactive")
        debt_dict = {}
    
    # Load or input CDS data
    print(f"\n[Step 3] Loading CDS spreads...")
    
    if args.cds_file:
        cds_dict = load_cds_from_csv(args.cds_file)
    elif args.interactive:
        cds_dict = get_manual_cds_input()
    else:
        print("Note: No CDS data provided. Use --cds-file or --interactive")
        cds_dict = {}
    
    # Process each company
    print(f"\n[Step 4] Processing companies...")
    
    results_list = []
    
    for ticker, stock_data in stock_data_dict.items():
        debt = debt_dict.get(ticker)
        market_cds = cds_dict.get(ticker)
        
        if debt is None:
            print(f"\n⚠ Skipping {ticker}: no debt data")
            continue
        
        result = process_real_company(
            ticker, stock_data, debt, market_cds,
            T=args.maturity,
            r=0.05,
            recovery=args.recovery
        )
        
        if result:
            results_list.append(result)
    
    # Create results dataframe
    if results_list:
        results_df = pd.DataFrame(results_list)
        
        # Display summary
        print(f"\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        print(results_df[['Company', 'Equity_Vol', 'Asset_Vol', 'Default_Prob', 
                          'Model_CDS_bps', 'Market_CDS_bps']].to_string(index=False))
        
        # Save results
        try:
            results_df.to_csv(args.output, index=False)
            print(f"\n✓ Results saved to: {args.output}")
        except Exception as e:
            print(f"\n✗ Failed to save results: {e}")
    else:
        print("\n✗ No results to display")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    """
    Usage examples:
    
    1. Fetch stock data + interactive debt input:
       python3 real_data.py --tickers AAPL MSFT TSLA --interactive
    
    2. Use CSV files for debt and CDS:
       python3 real_data.py --tickers AAPL MSFT TSLA --debt-file debt.csv --cds-file cds.csv
    
    3. Custom maturity and recovery:
       python3 real_data.py --tickers AAPL MSFT --debt-file debt.csv --maturity 5.0 --recovery 0.30
    
    Expected CSV format:
    
    debt.csv:
        ticker,total_debt
        AAPL,50
        MSFT,40
        TSLA,80
    
    cds.csv:
        ticker,cds_spread_bps
        AAPL,45
        MSFT,35
        TSLA,120
    """
    main()