"""
Merton Model Module: Core Structural Credit Model Functions

Computes:
- Default probability
- Asset value
- Distance to default
- Other Merton-related quantities
"""

import numpy as np
from scipy.stats import norm


class MertonModel:
    """
    Merton Structural Credit Model
    
    Models company default using asset dynamics:
    - Company assets V follow GBM with drift μ and volatility σ_V
    - Company defaults if V_T < D at maturity T
    - Equity = Call option on assets
    """
    
    def __init__(self, S, D, asset_vol, T, r, recovery_rate=0.40):
        """
        Initialize Merton model
        
        Args:
            S (float): Current stock price (equity value)
            D (float): Total debt (principal due at maturity)
            asset_vol (float): Asset volatility σ_V (calibrated from equity vol)
            T (float): Time to maturity (years)
            r (float): Risk-free rate (annualized)
            recovery_rate (float): Recovery rate in case of default (0-1)
        """
        self.S = S
        self.D = D
        self.sigma_V = asset_vol  # Asset volatility (calibrated)
        self.T = T
        self.r = r
        self.recovery_rate = recovery_rate
        
        # Compute asset value V from equity and debt
        # V = S + PV(D) ≈ S + D*e^(-rT)
        self.V = S + D * np.exp(-r * T)
        
        # Compute leverage ratio
        self.leverage = D / self.V
    
    def distance_to_default(self):
        """
        Compute Merton Distance to Default (DD)
        
        DD = [ln(V/D) + (r - 0.5*σ_V²)*T] / (σ_V*√T)
        
        This is similar to Black-Scholes d2 but for assets
        
        Returns:
            float: Distance to default
        """
        numerator = (
            np.log(self.V / self.D) + 
            (self.r - 0.5 * self.sigma_V**2) * self.T
        )
        denominator = self.sigma_V * np.sqrt(self.T)
        
        dd = numerator / denominator
        return dd
    
    def default_probability(self):
        """
        Compute risk-neutral probability of default
        
        PD = N(-DD) = N(-d2) in Merton model
        
        where N is the cumulative normal distribution
        
        Returns:
            float: Risk-neutral default probability (0-1)
        """
        dd = self.distance_to_default()
        pd = norm.cdf(-dd)  # N(-DD)
        return pd
    
    def loss_given_default(self):
        """
        Compute Loss Given Default (LGD)
        
        LGD = 1 - Recovery Rate
        
        Returns:
            float: LGD (0-1)
        """
        return 1.0 - self.recovery_rate
    
    def expected_loss(self):
        """
        Compute Expected Loss
        
        Expected Loss = PD × LGD × Notional
        
        For notional = 1 (unit principal):
        Expected Loss = PD × (1 - Recovery Rate)
        
        Returns:
            float: Expected loss per dollar of debt
        """
        pd = self.default_probability()
        lgd = self.loss_given_default()
        return pd * lgd
    
    def asset_value_at_maturity(self, num_paths=10000, seed=42):
        """
        Simulate asset value at maturity using Monte Carlo
        
        V_T = V * exp((r - 0.5*σ_V²)*T + σ_V*√T*Z)
        
        where Z ~ N(0,1)
        
        Args:
            num_paths (int): Number of Monte Carlo paths
            seed (int): Random seed for reproducibility
        
        Returns:
            np.array: Asset values at maturity (shape: num_paths,)
        """
        np.random.seed(seed)
        
        # Generate random normal shocks
        Z = np.random.standard_normal(num_paths)
        
        # Simulate asset value at maturity
        V_T = self.V * np.exp(
            (self.r - 0.5 * self.sigma_V**2) * self.T + 
            self.sigma_V * np.sqrt(self.T) * Z
        )
        
        return V_T
    
    def probability_of_default_monte_carlo(self, num_paths=10000):
        """
        Compute PD using Monte Carlo (for verification)
        
        Compare against analytical formula
        
        Args:
            num_paths (int): Number of simulation paths
        
        Returns:
            float: Monte Carlo estimate of PD
        """
        V_T = self.asset_value_at_maturity(num_paths=num_paths)
        
        # Count how many paths default (V_T < D)
        num_defaults = np.sum(V_T < self.D)
        
        # PD = fraction of paths that default
        pd_mc = num_defaults / num_paths
        
        return pd_mc
    
    def credit_spread(self):
        """
        Compute credit spread implied by Merton model
        
        Spread = -ln(1 - PD * LGD) / T
        
        This is the annual yield spread over risk-free rate
        
        Returns:
            float: Credit spread (annualized, as decimal)
        """
        pd = self.default_probability()
        lgd = self.loss_given_default()
        
        # Avoid log(0)
        survival_prob = 1.0 - pd * lgd
        
        if survival_prob <= 0:
            # Very high default risk
            return np.inf
        
        spread = -np.log(survival_prob) / self.T
        return spread
    
    def summary(self):
        """
        Print model summary
        """
        print("\n" + "=" * 70)
        print("MERTON MODEL SUMMARY")
        print("=" * 70)
        print(f"Stock Price (S):              ${self.S:.2f}")
        print(f"Debt (D):                     ${self.D:.2f}")
        print(f"Asset Value (V):              ${self.V:.2f}")
        print(f"Leverage (D/V):               {self.leverage:.2%}")
        print(f"Asset Volatility (σ_V):       {self.sigma_V:.2%}")
        print(f"Maturity (T):                 {self.T:.1f} years")
        print(f"Risk-free Rate (r):           {self.r:.2%}")
        print(f"Recovery Rate (R):            {self.recovery_rate:.2%}")
        print("-" * 70)
        print(f"Distance to Default (DD):     {self.distance_to_default():.4f}")
        print(f"Default Probability (PD):     {self.default_probability():.4f} ({self.default_probability():.2%})")
        print(f"Loss Given Default (LGD):     {self.loss_given_default():.2%}")
        print(f"Expected Loss:                {self.expected_loss():.4f}")
        print(f"Credit Spread:                {self.credit_spread():.2%} ({self.credit_spread()*10000:.0f} bps)")
        print("=" * 70 + "\n")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_d1_d2(V, D, sigma_V, T, r):
    """
    Compute Black-Scholes d1 and d2 for Merton model
    
    Args:
        V (float): Asset value
        D (float): Debt
        sigma_V (float): Asset volatility
        T (float): Maturity
        r (float): Risk-free rate
    
    Returns:
        tuple: (d1, d2)
    """
    d1 = (np.log(V/D) + (r + 0.5*sigma_V**2)*T) / (sigma_V*np.sqrt(T))
    d2 = d1 - sigma_V*np.sqrt(T)
    return d1, d2


def equity_value_from_merton(V, D, sigma_V, T, r):
    """
    Compute equity value using Merton formula
    
    S = V*N(d1) - D*e^(-rT)*N(d2)
    
    Args:
        V (float): Asset value
        D (float): Debt
        sigma_V (float): Asset volatility
        T (float): Maturity
        r (float): Risk-free rate
    
    Returns:
        float: Equity value
    """
    d1, d2 = compute_d1_d2(V, D, sigma_V, T, r)
    equity = V * norm.cdf(d1) - D * np.exp(-r*T) * norm.cdf(d2)
    return equity


# ============================================================================
# TEST CASES
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MERTON MODEL - TEST CASES")
    print("=" * 70)
    
    # Test 1: Basic scenario
    print("\nTest 1: Basic Scenario")
    print("-" * 70)
    
    S = 150.0
    D = 50.0
    asset_vol = 0.1898  # From calibration
    T = 1.0
    r = 0.05
    recovery_rate = 0.40
    
    model = MertonModel(S, D, asset_vol, T, r, recovery_rate)
    model.summary()
    
    # Test 2: High leverage
    print("\nTest 2: High Leverage (Risky) Company")
    print("-" * 70)
    
    model2 = MertonModel(
        S=100.0,
        D=100.0,
        asset_vol=0.205,  # From calibration
        T=1.0,
        r=0.05,
        recovery_rate=0.40
    )
    model2.summary()
    
    # Test 3: Low leverage
    print("\nTest 3: Low Leverage (Safe) Company")
    print("-" * 70)
    
    model3 = MertonModel(
        S=200.0,
        D=10.0,
        asset_vol=0.1909,
        T=1.0,
        r=0.05,
        recovery_rate=0.40
    )
    model3.summary()
    
    # Test 4: Verify Monte Carlo vs Analytical
    print("\nTest 4: Monte Carlo vs Analytical PD")
    print("-" * 70)
    
    model4 = MertonModel(S=150, D=50, asset_vol=0.20, T=1.0, r=0.05)
    
    pd_analytical = model4.default_probability()
    pd_mc = model4.probability_of_default_monte_carlo(num_paths=100000)
    
    print(f"Analytical PD:     {pd_analytical:.4f} ({pd_analytical:.2%})")
    print(f"Monte Carlo PD:    {pd_mc:.4f} ({pd_mc:.2%})")
    print(f"Difference:        {abs(pd_analytical - pd_mc):.4f}")
    print(f"Match? {np.isclose(pd_analytical, pd_mc, atol=0.01)}")
    
    # Test 5: Different recovery rates
    print("\nTest 5: Impact of Recovery Rate")
    print("-" * 70)
    
    recovery_rates = [0.20, 0.40, 0.60]
    
    print(f"{'Recovery':<12} {'PD':<12} {'LGD':<12} {'Expected Loss':<15}")
    print("-" * 70)
    
    for R in recovery_rates:
        model_r = MertonModel(S=150, D=50, asset_vol=0.20, T=1.0, r=0.05, recovery_rate=R)
        pd = model_r.default_probability()
        lgd = model_r.loss_given_default()
        el = model_r.expected_loss()
        print(f"{R:>11.0%}  {pd:>11.4f}  {lgd:>11.2%}  {el:>14.4f}")
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)