"""
CDS Pricing Module: Value Credit Default Swaps

A CDS (Credit Default Swap) is insurance against default:
- Buyer pays periodic premium (CDS spread) to seller
- Seller pays buyer if reference company defaults

This module prices CDS based on:
- Default probability (from Merton model)
- Recovery rate
- Maturity
- Risk-free rate
"""

import numpy as np
from scipy.stats import norm


def cds_spread_simple(default_probability, recovery_rate, notional=1.0):
    """
    Simple CDS spread formula
    
    CDS Spread = Default Probability × Loss Given Default × Notional
    
    where Loss Given Default (LGD) = 1 - Recovery Rate
    
    Args:
        default_probability (float): Risk-neutral PD (0-1)
        recovery_rate (float): Recovery rate in case of default (0-1)
        notional (float): Notional amount (default: $1)
    
    Returns:
        float: Annual CDS spread as fraction (e.g., 0.012 = 1.2% per year)
    """
    lgd = 1.0 - recovery_rate
    spread = default_probability * lgd * notional
    return spread


def cds_spread_bps(default_probability, recovery_rate, notional=1.0):
    """
    CDS spread in basis points (bps)
    
    1 bp = 0.01% = 0.0001 in decimal
    
    Args:
        default_probability (float): Risk-neutral PD (0-1)
        recovery_rate (float): Recovery rate (0-1)
        notional (float): Notional amount
    
    Returns:
        float: CDS spread in basis points
    
    Example:
        >>> cds_spread_bps(0.02, 0.40)
        120.0  # 120 basis points
    """
    spread_decimal = cds_spread_simple(default_probability, recovery_rate, notional)
    spread_bps = spread_decimal * 10000
    return spread_bps


def cds_fair_spread(default_probability, recovery_rate, T, r, notional=1.0):
    """
    More sophisticated CDS pricing using annuity factor
    
    Fair CDS Spread = PV(Default Loss) / PV(Annuity of Premiums)
    
    Simplified:
    Spread = λ × LGD / A(T, r)
    
    where:
    - λ = default intensity (hazard rate)
    - LGD = loss given default
    - A(T, r) = annuity factor
    
    Args:
        default_probability (float): Risk-neutral PD for maturity T
        recovery_rate (float): Recovery rate (0-1)
        T (float): Maturity in years
        r (float): Risk-free rate (annualized)
        notional (float): Notional amount
    
    Returns:
        float: Fair CDS spread (as decimal, e.g., 0.012 for 1.2%)
    """
    # Extract hazard rate λ from PD
    # For constant hazard rate: PD = 1 - exp(-λ*T)
    # So: λ = -ln(1 - PD) / T
    
    if default_probability >= 1.0:
        # Certain default
        return np.inf
    
    hazard_rate = -np.log(1.0 - default_probability) / T
    
    # Annuity factor = PV of $1 received annually for T years
    # A = (1 - exp(-r*T)) / r
    if abs(r) < 1e-6:
        # If r ≈ 0, annuity factor ≈ T
        annuity_factor = T
    else:
        annuity_factor = (1.0 - np.exp(-r * T)) / r
    
    # LGD
    lgd = 1.0 - recovery_rate
    
    # Fair spread
    spread = (hazard_rate * lgd / annuity_factor) * notional
    
    return spread


def cds_fair_spread_bps(default_probability, recovery_rate, T, r, notional=1.0):
    """
    Fair CDS spread in basis points
    
    Args:
        default_probability (float): Risk-neutral PD
        recovery_rate (float): Recovery rate (0-1)
        T (float): Maturity (years)
        r (float): Risk-free rate
        notional (float): Notional amount
    
    Returns:
        float: Fair CDS spread in basis points
    """
    spread_decimal = cds_fair_spread(default_probability, recovery_rate, T, r, notional)
    spread_bps = spread_decimal * 10000
    return spread_bps


def cds_pv_premium_leg(cds_spread, T, r, notional=1.0, payment_freq=4):
    """
    Present value of premium payments (what buyer pays)
    
    Buyer pays CDS spread periodically (e.g., quarterly) until maturity or default
    
    Args:
        cds_spread (float): CDS spread (as decimal, e.g., 0.012)
        T (float): Maturity (years)
        r (float): Risk-free rate
        notional (float): Notional amount
        payment_freq (int): Payment frequency per year (default: 4 = quarterly)
    
    Returns:
        float: PV of premium leg
    """
    # Payment dates
    dt = 1.0 / payment_freq  # Time between payments
    num_payments = int(payment_freq * T)
    
    pv_premium = 0.0
    
    for i in range(1, num_payments + 1):
        t = i * dt
        
        # Discount factor
        df = np.exp(-r * t)
        
        # Accrued premium payment
        premium_payment = cds_spread * notional * dt
        
        # PV of this payment
        pv_premium += premium_payment * df
    
    return pv_premium


def cds_pv_protection_leg(default_probability, recovery_rate, T, r, notional=1.0):
    """
    Present value of protection payments (what seller pays if default)
    
    Seller pays (1 - Recovery) × Notional if default occurs
    
    Simplified: PV = PD × (1 - R) × Notional × Discount Factor
    
    Args:
        default_probability (float): Risk-neutral PD
        recovery_rate (float): Recovery rate (0-1)
        T (float): Maturity (years)
        r (float): Risk-free rate
        notional (float): Notional amount
    
    Returns:
        float: PV of protection leg
    """
    # Simplified: assume default happens at mid-point of maturity
    # More sophisticated: integrate over time
    
    loss_given_default = (1.0 - recovery_rate) * notional
    
    # Assume expected default time = T/2
    discount_factor = np.exp(-r * T / 2)
    
    pv_protection = default_probability * loss_given_default * discount_factor
    
    return pv_protection


def cds_price_from_spread(market_spread, fair_spread, T, r, notional=1.0, payment_freq=4):
    """
    Compute CDS value (profit/loss) if market spread differs from fair spread
    
    Value = PV(Protection Leg) - PV(Premium Leg @ market spread)
    
    If market_spread > fair_spread, CDS is expensive (seller wins)
    If market_spread < fair_spread, CDS is cheap (buyer wins)
    
    Args:
        market_spread (float): CDS spread observed in market
        fair_spread (float): Theoretically fair CDS spread
        T (float): Maturity
        r (float): Risk-free rate
        notional (float): Notional amount
        payment_freq (int): Payment frequency per year
    
    Returns:
        float: CDS value (positive = profitable for seller, negative = profitable for buyer)
    """
    # PV of premium leg at market spread
    pv_premium_market = cds_pv_premium_leg(market_spread, T, r, notional, payment_freq)
    
    # PV of premium leg at fair spread
    pv_premium_fair = cds_pv_premium_leg(fair_spread, T, r, notional, payment_freq)
    
    # Value to seller = fair premium PV - market premium PV
    # If positive, seller is overpaid (good for seller)
    value_to_seller = pv_premium_fair - pv_premium_market
    
    return value_to_seller


class CDSContract:
    """
    Represents a CDS contract
    """
    
    def __init__(self, reference_entity, notional, maturity, recovery_rate,
                 risk_free_rate, default_probability, payment_freq=4):
        """
        Initialize CDS contract
        
        Args:
            reference_entity (str): Name of company being insured (e.g., "Apple")
            notional (float): Notional amount of protection
            maturity (float): Maturity in years
            recovery_rate (float): Recovery rate in case of default (0-1)
            risk_free_rate (float): Risk-free rate (annualized)
            default_probability (float): Risk-neutral default probability
            payment_freq (int): Premium payment frequency per year
        """
        self.reference_entity = reference_entity
        self.notional = notional
        self.T = maturity
        self.recovery_rate = recovery_rate
        self.r = risk_free_rate
        self.pd = default_probability
        self.payment_freq = payment_freq
    
    def fair_spread(self):
        """Compute fair (theoretical) CDS spread"""
        return cds_fair_spread(self.pd, self.recovery_rate, self.T, self.r, self.notional)
    
    def fair_spread_bps(self):
        """Compute fair CDS spread in basis points"""
        return cds_fair_spread_bps(self.pd, self.recovery_rate, self.T, self.r, self.notional)
    
    def simple_spread_bps(self):
        """Compute simple CDS spread in basis points"""
        return cds_spread_bps(self.pd, self.recovery_rate, self.notional)
    
    def pv_premium_leg(self, spread):
        """Compute PV of premium leg at given spread"""
        return cds_pv_premium_leg(spread, self.T, self.r, self.notional, self.payment_freq)
    
    def pv_protection_leg(self):
        """Compute PV of protection leg"""
        return cds_pv_protection_leg(self.pd, self.recovery_rate, self.T, self.r, self.notional)
    
    def summary(self):
        """Print contract summary"""
        fair_spread = self.fair_spread_bps()
        simple_spread = self.simple_spread_bps()
        
        print("\n" + "=" * 70)
        print("CDS CONTRACT SUMMARY")
        print("=" * 70)
        print(f"Reference Entity:             {self.reference_entity}")
        print(f"Notional Amount:              ${self.notional:,.2f}")
        print(f"Maturity:                     {self.T:.1f} years")
        print(f"Recovery Rate:                {self.recovery_rate:.2%}")
        print(f"Risk-free Rate:               {self.r:.2%}")
        print(f"Default Probability (PD):     {self.pd:.4f} ({self.pd:.2%})")
        print(f"Payment Frequency:            {self.payment_freq}x per year")
        print("-" * 70)
        print(f"Fair CDS Spread (simple):     {simple_spread:.1f} bps")
        print(f"Fair CDS Spread (annuity):    {fair_spread:.1f} bps")
        print(f"PV of Premium Leg (@fair):    ${self.pv_premium_leg(self.fair_spread()):,.2f}")
        print(f"PV of Protection Leg:         ${self.pv_protection_leg():,.2f}")
        print("=" * 70 + "\n")


# ============================================================================
# TEST CASES
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CDS PRICING - TEST CASES")
    print("=" * 70)
    
    # Test 1: Simple CDS pricing
    print("\nTest 1: Simple CDS Pricing")
    print("-" * 70)
    
    pd = 0.02  # 2% default probability
    recovery = 0.40  # 40% recovery
    
    spread_simple = cds_spread_simple(pd, recovery)
    spread_bps = cds_spread_bps(pd, recovery)
    
    print(f"Default Probability:  {pd:.2%}")
    print(f"Recovery Rate:        {recovery:.2%}")
    print(f"CDS Spread (decimal): {spread_simple:.4f} ({spread_simple:.2%})")
    print(f"CDS Spread (bps):     {spread_bps:.1f} bps")
    print(f"Interpretation: Buyer pays {spread_bps:.0f} bps/year for protection")
    
    # Test 2: CDS contract with full details
    print("\nTest 2: Full CDS Contract")
    print("-" * 70)
    
    cds = CDSContract(
        reference_entity="Apple Inc.",
        notional=1_000_000,  # $1 million notional
        maturity=5.0,  # 5-year CDS
        recovery_rate=0.40,
        risk_free_rate=0.05,
        default_probability=0.02,
        payment_freq=4  # Quarterly payments
    )
    
    cds.summary()
    
    # Test 3: Different recovery rates
    print("\nTest 3: Impact of Recovery Rate on CDS Spread")
    print("-" * 70)
    
    print(f"{'Recovery':<12} {'Simple (bps)':<15} {'Fair (bps)':<15}")
    print("-" * 70)
    
    recovery_rates = [0.10, 0.30, 0.50, 0.70]
    
    for r in recovery_rates:
        cds_simple = cds_spread_bps(0.02, r)
        cds_fair = cds_fair_spread_bps(0.02, r, T=5.0, r=0.05)
        print(f"{r:>11.0%}  {cds_simple:>14.1f}  {cds_fair:>14.1f}")
    
    # Test 4: Different default probabilities
    print("\nTest 4: Impact of Default Probability on CDS Spread")
    print("-" * 70)
    
    print(f"{'PD':<12} {'Simple (bps)':<15} {'Fair (bps)':<15}")
    print("-" * 70)
    
    pds = [0.005, 0.010, 0.020, 0.050, 0.100]
    
    for pd in pds:
        cds_simple = cds_spread_bps(pd, 0.40)
        cds_fair = cds_fair_spread_bps(pd, 0.40, T=5.0, r=0.05)
        print(f"{pd:>11.2%}  {cds_simple:>14.1f}  {cds_fair:>14.1f}")
    
    # Test 5: Fair spread vs market spread
    print("\nTest 5: CDS Value if Market Spread ≠ Fair Spread")
    print("-" * 70)
    
    fair_spread = cds_fair_spread(0.02, 0.40, T=5.0, r=0.05, notional=1e6)
    fair_spread_bps = fair_spread * 10000
    
    print(f"Fair Spread: {fair_spread_bps:.1f} bps")
    print(f"\n{'Market Spread (bps)':<20} {'Value to Seller':<20}")
    print("-" * 70)
    
    market_spreads_bps = [100, 120, 150, 180, 200]
    
    for mkt_bps in market_spreads_bps:
        mkt_spread = mkt_bps / 10000
        value = cds_price_from_spread(mkt_spread, fair_spread, T=5.0, r=0.05, notional=1e6)
        sign = "+" if value > 0 else "-"
        print(f"{mkt_bps:>19.0f}  {sign}${abs(value):>18,.0f}")
    
    print("\n(Positive = good for seller, Negative = good for buyer)")
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)