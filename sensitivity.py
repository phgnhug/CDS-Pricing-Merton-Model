"""
Sensitivity Analysis Module: Greeks and Parameter Sensitivity

Compute how CDS spreads change when market parameters change:
- Delta Volatility (Vega): ∂CDS / ∂σ_V
- Delta Leverage: ∂CDS / ∂(D/V)
- Delta Recovery: ∂CDS / ∂R
- Gamma: Second derivative (convexity)

Uses finite difference method (bump-and-recompute)
"""

import numpy as np
from merton_model import MertonModel
from cds_pricing import cds_spread_bps, cds_fair_spread_bps


class CDSSensitivity:
    """
    Compute sensitivity of CDS spreads to model parameters
    """
    
    def __init__(self, S, D, asset_vol, T, r, recovery_rate):
        """
        Initialize sensitivity analysis
        
        Args:
            S (float): Stock price
            D (float): Debt
            asset_vol (float): Asset volatility
            T (float): Maturity
            r (float): Risk-free rate
            recovery_rate (float): Recovery rate (0-1)
        """
        self.S = S
        self.D = D
        self.sigma_V = asset_vol
        self.T = T
        self.r = r
        self.recovery_rate = recovery_rate
    
    def _compute_cds_spread(self, sigma_V=None, recovery=None):
        """
        Compute CDS spread with given parameters
        
        Args:
            sigma_V (float): Asset volatility (use self.sigma_V if None)
            recovery (float): Recovery rate (use self.recovery_rate if None)
        
        Returns:
            float: CDS spread in bps
        """
        sv = sigma_V if sigma_V is not None else self.sigma_V
        r = recovery if recovery is not None else self.recovery_rate
        
        # Create Merton model
        model = MertonModel(self.S, self.D, sv, self.T, self.r, r)
        
        # Get default probability
        pd = model.default_probability()
        
        # Compute fair CDS spread
        spread_bps = cds_fair_spread_bps(pd, r, self.T, self.r)
        
        return spread_bps
    
    def delta_volatility(self, bump=0.001, method='central'):
        """
        Compute delta with respect to asset volatility (Vega)
        
        Vega = ∂CDS / ∂σ_V
        
        Args:
            bump (float): Bump size (default: 0.1% = 0.001)
            method (str): 'central' or 'forward' difference
        
        Returns:
            float: Change in CDS (bps) per 1% change in volatility
        """
        if method == 'central':
            # Central difference: (f(x+h) - f(x-h)) / (2h)
            cds_up = self._compute_cds_spread(sigma_V=self.sigma_V + bump)
            cds_down = self._compute_cds_spread(sigma_V=self.sigma_V - bump)
            vega = (cds_up - cds_down) / (2 * bump)
        else:
            # Forward difference: (f(x+h) - f(x)) / h
            cds_base = self._compute_cds_spread()
            cds_up = self._compute_cds_spread(sigma_V=self.sigma_V + bump)
            vega = (cds_up - cds_base) / bump
        
        return vega
    
    def delta_recovery(self, bump=0.01, method='central'):
        """
        Compute delta with respect to recovery rate
        
        ∂CDS / ∂R (note: negative, because higher recovery → lower CDS)
        
        Args:
            bump (float): Bump size (default: 1% = 0.01)
            method (str): 'central' or 'forward' difference
        
        Returns:
            float: Change in CDS (bps) per 1% change in recovery
        """
        if method == 'central':
            cds_up = self._compute_cds_spread(recovery=self.recovery_rate + bump)
            cds_down = self._compute_cds_spread(recovery=self.recovery_rate - bump)
            delta_r = (cds_up - cds_down) / (2 * bump)
        else:
            cds_base = self._compute_cds_spread()
            cds_up = self._compute_cds_spread(recovery=self.recovery_rate + bump)
            delta_r = (cds_up - cds_base) / bump
        
        return delta_r
    
    def gamma_volatility(self, bump=0.001):
        """
        Compute gamma (second derivative) with respect to volatility
        
        Gamma = ∂²CDS / ∂σ_V²
        
        Measures convexity: how much delta changes when vol changes
        
        Args:
            bump (float): Bump size
        
        Returns:
            float: Gamma (convexity)
        """
        cds_base = self._compute_cds_spread()
        cds_up = self._compute_cds_spread(sigma_V=self.sigma_V + bump)
        cds_down = self._compute_cds_spread(sigma_V=self.sigma_V - bump)
        
        # Second derivative: (f(x+h) - 2f(x) + f(x-h)) / h²
        gamma = (cds_up - 2*cds_base + cds_down) / (bump**2)
        
        return gamma
    
    def sensitivity_table(self, param_name, param_values, param_key):
        """
        Create sensitivity table for a given parameter
        
        Args:
            param_name (str): Parameter name (e.g., "Volatility")
            param_values (array): Values to test
            param_key (str): 'sigma_V' or 'recovery'
        
        Returns:
            dict: {value: cds_spread}
        """
        results = {}
        
        for val in param_values:
            if param_key == 'sigma_V':
                cds = self._compute_cds_spread(sigma_V=val)
            elif param_key == 'recovery':
                cds = self._compute_cds_spread(recovery=val)
            else:
                raise ValueError(f"Unknown parameter: {param_key}")
            
            results[val] = cds
        
        return results
    
    def summary(self):
        """
        Print sensitivity summary
        """
        base_spread = self._compute_cds_spread()
        vega = self.delta_volatility()
        delta_r = self.delta_recovery()
        gamma = self.gamma_volatility()
        
        print("\n" + "=" * 70)
        print("CDS SENSITIVITY SUMMARY")
        print("=" * 70)
        print(f"Base CDS Spread:              {base_spread:.1f} bps")
        print("-" * 70)
        print(f"Vega (∂CDS/∂σ_V):             {vega:.2f} bps per 1% vol change")
        print(f"Delta Recovery (∂CDS/∂R):     {delta_r:.2f} bps per 1% recovery change")
        print(f"Gamma (∂²CDS/∂σ_V²):          {gamma:.4f} bps/vol² (convexity)")
        print("=" * 70 + "\n")
        
        print("Interpretation:")
        print(f"- If volatility increases 1%, CDS spread changes by {vega:.2f} bps")
        print(f"- If recovery rate increases 1%, CDS spread changes by {delta_r:.2f} bps")
        print(f"- Gamma = {gamma:.4f} means CDS has {'positive' if gamma > 0 else 'negative'} convexity")
        print()


def compute_sensitivity_matrix(S, D, asset_vol, T, r, recovery,
                               vol_range, recovery_range):
    """
    Compute CDS spread across a grid of parameters
    
    Args:
        S, D, asset_vol, T, r, recovery (float): Base parameters
        vol_range (array): Asset volatility values to test
        recovery_range (array): Recovery rate values to test
    
    Returns:
        tuple: (vol_range, recovery_range, matrix)
        where matrix[i,j] = CDS spread for vol[i], recovery[j]
    """
    matrix = np.zeros((len(vol_range), len(recovery_range)))
    
    for i, sv in enumerate(vol_range):
        for j, rec in enumerate(recovery_range):
            model = MertonModel(S, D, sv, T, r, rec)
            pd = model.default_probability()
            cds = cds_fair_spread_bps(pd, rec, T, r)
            matrix[i, j] = cds
    
    return vol_range, recovery_range, matrix


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CDS SENSITIVITY - TEST CASES")
    print("=" * 70)
    
    # Base parameters
    S = 150.0
    D = 50.0
    asset_vol = 0.20
    T = 5.0
    r = 0.05
    recovery = 0.40
    
    sensitivity = CDSSensitivity(S, D, asset_vol, T, r, recovery)
    
    # Test 1: Basic sensitivity metrics
    print("\nTest 1: Sensitivity Metrics (Greeks)")
    print("-" * 70)
    sensitivity.summary()
    
    # Test 2: Volatility sensitivity
    print("\nTest 2: CDS Spread vs Asset Volatility")
    print("-" * 70)
    
    vol_values = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
    vol_sensitivity = sensitivity.sensitivity_table("Volatility", vol_values, 'sigma_V')
    
    print(f"{'Asset Vol':<15} {'CDS Spread (bps)':<20}")
    print("-" * 70)
    for vol, cds in vol_sensitivity.items():
        print(f"{vol:>14.0%}  {cds:>19.1f}")
    
    # Test 3: Recovery sensitivity
    print("\nTest 3: CDS Spread vs Recovery Rate")
    print("-" * 70)
    
    recovery_values = np.array([0.20, 0.30, 0.40, 0.50, 0.60])
    recovery_sensitivity = sensitivity.sensitivity_table("Recovery", recovery_values, 'recovery')
    
    print(f"{'Recovery':<15} {'CDS Spread (bps)':<20}")
    print("-" * 70)
    for rec, cds in recovery_sensitivity.items():
        print(f"{rec:>14.0%}  {cds:>19.1f}")
    
    # Test 4: 2D sensitivity matrix
    print("\nTest 4: 2D Sensitivity Matrix (Vol × Recovery)")
    print("-" * 70)
    
    vol_range = np.array([0.15, 0.20, 0.25, 0.30])
    recovery_range = np.array([0.30, 0.40, 0.50, 0.60])
    
    vol_grid, rec_grid, matrix = compute_sensitivity_matrix(
        S, D, asset_vol, T, r, recovery,
        vol_range, recovery_range
    )
    
    print("\nCDS Spread (bps) - Rows: Asset Vol, Columns: Recovery Rate")
    print("-" * 70)
    
    # Print header
    header = "Vol/Rec  "
    for rec in rec_grid:
        header += f"{rec:>12.0%}"
    print(header)
    print("-" * 70)
    
    # Print matrix
    for i, vol in enumerate(vol_grid):
        row = f"{vol:>7.0%}  "
        for j, rec in enumerate(rec_grid):
            row += f"{matrix[i,j]:>12.1f}"
        print(row)
    
    # Test 5: Stress testing
    print("\nTest 5: Stress Test - CDS Spread Under Different Scenarios")
    print("-" * 70)
    
    scenarios = {
        'Base Case': {'vol': 0.20, 'recovery': 0.40},
        'High Vol': {'vol': 0.35, 'recovery': 0.40},
        'Low Recovery': {'vol': 0.20, 'recovery': 0.25},
        'High Risk': {'vol': 0.35, 'recovery': 0.25},
        'Safe': {'vol': 0.15, 'recovery': 0.60},
    }
    
    print(f"{'Scenario':<15} {'Asset Vol':<15} {'Recovery':<15} {'CDS (bps)':<15}")
    print("-" * 70)
    
    for scenario_name, params in scenarios.items():
        model = MertonModel(S, D, params['vol'], T, r, params['recovery'])
        pd = model.default_probability()
        cds = cds_fair_spread_bps(pd, params['recovery'], T, r)
        print(f"{scenario_name:<15} {params['vol']:>14.0%} {params['recovery']:>14.0%} {cds:>14.1f}")
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)