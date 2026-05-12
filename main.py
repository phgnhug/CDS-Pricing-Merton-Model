"""
Main Pipeline: Complete CDS Pricing Workflow

Flow:
1. Load/define company data
2. Calibrate Merton model (extract asset volatility)
3. Compute default probability
4. Price CDS
5. Calculate sensitivity (Greeks)
6. Generate results and visualizations
7. Save to CSV
"""

import numpy as np
import pandas as pd
from calibration import calibrate_merton
from merton_model import MertonModel
from cds_pricing import cds_fair_spread_bps, CDSContract
from sensitivity import CDSSensitivity


# ============================================================================
# SAMPLE DATA: Companies with market data
# ============================================================================

COMPANY_DATA = {
    'Apple': {
        'stock_price': 150.0,
        'equity_vol': 0.25,
        'total_debt': 50.0,
        'market_cds_bps': 45,  # Market CDS spread for reference
    },
    'Microsoft': {
        'stock_price': 300.0,
        'equity_vol': 0.20,
        'total_debt': 40.0,
        'market_cds_bps': 35,
    },
    'Tesla': {
        'stock_price': 200.0,
        'equity_vol': 0.35,
        'total_debt': 80.0,
        'market_cds_bps': 120,
    },
    'Goldman Sachs': {
        'stock_price': 320.0,
        'equity_vol': 0.30,
        'total_debt': 200.0,
        'market_cds_bps': 80,
    },
    'Risky Corp': {
        'stock_price': 50.0,
        'equity_vol': 0.50,
        'total_debt': 80.0,
        'market_cds_bps': 300,
    },
}

# Global parameters
MATURITY = 1.0  # 1-year CDS
RISK_FREE_RATE = 0.05  # 5%
RECOVERY_RATE = 0.40  # 40% recovery
OUTPUT_FILE = 'data/results.csv'


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def process_company(name, data, T, r, recovery):
    """
    Process one company: calibrate, price, analyze
    
    Args:
        name (str): Company name
        data (dict): Market data
        T (float): Maturity
        r (float): Risk-free rate
        recovery (float): Recovery rate
    
    Returns:
        dict: Results
    """
    print(f"\n{'='*70}")
    print(f"Processing: {name}")
    print(f"{'='*70}")
    
    S = data['stock_price']
    equity_vol = data['equity_vol']
    D = data['total_debt']
    market_cds = data['market_cds_bps']
    
    # Step 1: Calibration
    print(f"\n[1] Calibration")
    print(f"    Stock price: ${S:.2f}")
    print(f"    Equity vol: {equity_vol:.2%}")
    print(f"    Total debt: ${D:.2f}")
    
    try:
        asset_vol = calibrate_merton(S, equity_vol, D, T, r)
        print(f"    ✓ Asset vol (calibrated): {asset_vol:.2%}")
    except Exception as e:
        print(f"    ✗ Calibration failed: {e}")
        return None
    
    # Step 2: Merton Model
    print(f"\n[2] Merton Model")
    model = MertonModel(S, D, asset_vol, T, r, recovery)
    
    dd = model.distance_to_default()
    pd = model.default_probability()
    
    print(f"    Distance to Default: {dd:.4f}")
    print(f"    Default Probability: {pd:.4f} ({pd:.2%})")
    
    # Step 3: CDS Pricing
    print(f"\n[3] CDS Pricing")
    cds_spread = cds_fair_spread_bps(pd, recovery, T, r)
    
    print(f"    Model CDS Spread: {cds_spread:.1f} bps")
    print(f"    Market CDS Spread: {market_cds:.1f} bps")
    print(f"    Difference: {cds_spread - market_cds:.1f} bps")
    
    if cds_spread > market_cds:
        print(f"    → Model overprices CDS (sell overvalued)")
    else:
        print(f"    → Model underprices CDS (buy undervalued)")
    
    # Step 4: Sensitivity Analysis
    print(f"\n[4] Sensitivity Analysis (Greeks)")
    sensitivity = CDSSensitivity(S, D, asset_vol, T, r, recovery)
    
    vega = sensitivity.delta_volatility()
    delta_r = sensitivity.delta_recovery()
    gamma = sensitivity.gamma_volatility()
    
    print(f"    Vega (∂CDS/∂σ_V): {vega:.2f} bps per 1% vol")
    print(f"    Delta Recovery: {delta_r:.2f} bps per 1% recovery")
    print(f"    Gamma (convexity): {gamma:.4f}")
    
    # Compile results
    results = {
        'Company': name,
        'Stock_Price': S,
        'Equity_Vol': equity_vol,
        'Total_Debt': D,
        'Leverage': D / (S + D * np.exp(-r * T)),
        'Asset_Vol': asset_vol,
        'Distance_to_Default': dd,
        'Default_Prob': pd,
        'Model_CDS_bps': cds_spread,
        'Market_CDS_bps': market_cds,
        'CDS_Difference': cds_spread - market_cds,
        'Vega': vega,
        'Delta_Recovery': delta_r,
        'Gamma': gamma,
    }
    
    print(f"\n✓ {name} processed successfully")
    
    return results


def main():
    """
    Main pipeline: process all companies and generate results
    """
    print("\n" + "="*70)
    print("CDS PRICING WITH MERTON STRUCTURAL MODEL")
    print("="*70)
    print(f"Maturity: {MATURITY:.1f} years")
    print(f"Risk-free rate: {RISK_FREE_RATE:.2%}")
    print(f"Recovery rate: {RECOVERY_RATE:.2%}")
    
    # Process each company
    all_results = []
    
    for company_name, company_data in COMPANY_DATA.items():
        result = process_company(
            company_name,
            company_data,
            T=MATURITY,
            r=RISK_FREE_RATE,
            recovery=RECOVERY_RATE
        )
        
        if result is not None:
            all_results.append(result)
    
    # Create results dataframe
    results_df = pd.DataFrame(all_results)
    
    # Display summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE: ALL COMPANIES")
    print("="*70)
    
    # Select key columns for display
    display_cols = [
        'Company',
        'Equity_Vol',
        'Asset_Vol',
        'Default_Prob',
        'Model_CDS_bps',
        'Market_CDS_bps',
        'CDS_Difference'
    ]
    
    print("\n" + results_df[display_cols].to_string(index=False))
    
    # Statistics
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    
    print(f"\nNumber of companies analyzed: {len(results_df)}")
    print(f"\nModel CDS Spreads:")
    print(f"  Min: {results_df['Model_CDS_bps'].min():.1f} bps")
    print(f"  Max: {results_df['Model_CDS_bps'].max():.1f} bps")
    print(f"  Mean: {results_df['Model_CDS_bps'].mean():.1f} bps")
    print(f"  Std Dev: {results_df['Model_CDS_bps'].std():.1f} bps")
    
    print(f"\nModel vs Market CDS Spreads:")
    print(f"  Mean absolute error: {results_df['CDS_Difference'].abs().mean():.1f} bps")
    print(f"  Correlation: {results_df['Model_CDS_bps'].corr(results_df['Market_CDS_bps']):.4f}")
    
    print(f"\nDefault Probabilities:")
    print(f"  Min: {results_df['Default_Prob'].min():.4f} ({results_df['Default_Prob'].min():.2%})")
    print(f"  Max: {results_df['Default_Prob'].max():.4f} ({results_df['Default_Prob'].max():.2%})")
    print(f"  Mean: {results_df['Default_Prob'].mean():.4f} ({results_df['Default_Prob'].mean():.2%})")
    
    print(f"\nVega (sensitivity to volatility):")
    print(f"  Mean: {results_df['Vega'].mean():.2f} bps per 1% vol")
    print(f"  Range: [{results_df['Vega'].min():.2f}, {results_df['Vega'].max():.2f}]")
    
    # Save to CSV
    try:
        results_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✓ Results saved to: {OUTPUT_FILE}")
    except Exception as e:
        print(f"\n✗ Failed to save results: {e}")
    
    # Analysis: Which companies are mispriced?
    print("\n" + "="*70)
    print("PRICING ANALYSIS")
    print("="*70)
    
    overpriced = results_df[results_df['CDS_Difference'] > 0]
    underpriced = results_df[results_df['CDS_Difference'] < 0]
    
    if len(overpriced) > 0:
        print(f"\nOVERPRICED (Model > Market) - SELL CDS:")
        for _, row in overpriced.iterrows():
            print(f"  {row['Company']}: {row['CDS_Difference']:+.1f} bps " 
                  f"({row['Model_CDS_bps']:.1f} vs {row['Market_CDS_bps']:.1f})")
    
    if len(underpriced) > 0:
        print(f"\nUNDERPRICED (Model < Market) - BUY CDS:")
        for _, row in underpriced.iterrows():
            print(f"  {row['Company']}: {row['CDS_Difference']:.1f} bps " 
                  f"({row['Model_CDS_bps']:.1f} vs {row['Market_CDS_bps']:.1f})")
    
    # Risk analysis
    print("\n" + "="*70)
    print("RISK PROFILE")
    print("="*70)
    
    results_df['Risk_Level'] = pd.cut(
        results_df['Default_Prob'],
        bins=[0, 0.01, 0.05, 0.10, 1.0],
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    print("\nCompanies by risk level:")
    for risk_level in ['Low', 'Medium', 'High', 'Very High']:
        companies = results_df[results_df['Risk_Level'] == risk_level]['Company'].tolist()
        if companies:
            print(f"  {risk_level}: {', '.join(companies)}")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    
    return results_df


# ============================================================================
# HELPER FUNCTIONS FOR ADVANCED ANALYSIS
# ============================================================================

def sensitivity_report(company_name, data, T, r, recovery):
    """
    Generate detailed sensitivity report for one company
    """
    print(f"\nDETAILED SENSITIVITY REPORT: {company_name}")
    print("="*70)
    
    S = data['stock_price']
    equity_vol = data['equity_vol']
    D = data['total_debt']
    
    # Calibrate
    asset_vol = calibrate_merton(S, equity_vol, D, T, r)
    
    # Sensitivity analysis
    sensitivity = CDSSensitivity(S, D, asset_vol, T, r, recovery)
    sensitivity.summary()


def stress_test(company_name, data, T, r, recovery, vol_shock, recovery_shock):
    """
    Stress test a company under parameter shocks
    
    Args:
        vol_shock (float): Volatility shock (e.g., 0.10 for +10%)
        recovery_shock (float): Recovery shock (e.g., -0.10 for -10%)
    """
    print(f"\nSTRESS TEST: {company_name}")
    print("="*70)
    
    S = data['stock_price']
    equity_vol = data['equity_vol']
    D = data['total_debt']
    
    # Base case
    asset_vol = calibrate_merton(S, equity_vol, D, T, r)
    model = MertonModel(S, D, asset_vol, T, r, recovery)
    pd_base = model.default_probability()
    cds_base = cds_fair_spread_bps(pd_base, recovery, T, r)
    
    # Stressed case
    asset_vol_stressed = asset_vol + vol_shock
    model_stressed = MertonModel(S, D, asset_vol_stressed, T, r, recovery + recovery_shock)
    pd_stressed = model_stressed.default_probability()
    cds_stressed = cds_fair_spread_bps(pd_stressed, recovery + recovery_shock, T, r)
    
    print(f"Base Case:")
    print(f"  Asset Vol: {asset_vol:.2%}, Default Prob: {pd_base:.2%}, CDS: {cds_base:.1f} bps")
    print(f"\nStress Scenario (+{vol_shock:.0%} vol, {recovery_shock:+.0%} recovery):")
    print(f"  Asset Vol: {asset_vol_stressed:.2%}, Default Prob: {pd_stressed:.2%}, CDS: {cds_stressed:.1f} bps")
    print(f"\nChange: CDS +{cds_stressed - cds_base:.1f} bps ({(cds_stressed/cds_base - 1)*100:+.1f}%)")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run main pipeline
    results = main()
    
    # Optional: Run detailed analysis on specific companies
    print("\n" + "="*70)
    print("OPTIONAL: DETAILED ANALYSIS")
    print("="*70)
    
    # Example: Sensitivity report for Tesla
    sensitivity_report('Tesla', COMPANY_DATA['Tesla'], MATURITY, RISK_FREE_RATE, RECOVERY_RATE)
    
    # Example: Stress test for Risky Corp
    stress_test('Risky Corp', COMPANY_DATA['Risky Corp'], 
                MATURITY, RISK_FREE_RATE, RECOVERY_RATE,
                vol_shock=0.10, recovery_shock=-0.10)
    
    print("\n✓ All analysis complete! Check 'data/results.csv' for full results.")