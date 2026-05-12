"""
Calibration Module: Extract Asset Volatility from Equity Market Data

Using Newton-Raphson method to solve the inverse problem:
Given: Stock price S, Equity volatility σ_S, Debt D, Maturity T, Risk-free rate r
Find: Asset volatility σ_V

Based on Black-Scholes relationship:
σ_S * S = σ_V * V * N(d1)

where V = S + PV(D) and d1 is the standard Merton d1 formula
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import newton


class MertonCalibration:
    """Calibrate Merton model to extract asset volatility"""
    
    def __init__(self, S, equity_vol, D, T, r):
        """
        Initialize calibration problem
        
        Args:
            S (float): Current stock price
            equity_vol (float): Equity volatility (annualized)
            D (float): Total debt value
            T (float): Time to maturity (in years)
            r (float): Risk-free rate (annualized)
        """
        self.S = S  # Stock price
        self.sigma_S = equity_vol  # Equity volatility
        self.D = D  # Debt
        self.T = T  # Maturity
        self.r = r  # Risk-free rate
        
        # Initial guess: leverage-adjusted equity vol
        # Rough approximation: asset_vol ≈ equity_vol * S / (S + D)
        self.initial_guess = equity_vol * S / (S + D)
    
    def _black_scholes_d1(self, V, sigma_V):
        """
        Compute d1 in Black-Scholes formula
        
        d1 = [ln(V/D) + (r + 0.5*sigma_V^2)*T] / (sigma_V * sqrt(T))
        """
        numerator = np.log(V / self.D) + (self.r + 0.5 * sigma_V**2) * self.T
        denominator = sigma_V * np.sqrt(self.T)
        return numerator / denominator
    
    def _black_scholes_call(self, V, sigma_V):
        """
        Compute equity value using Black-Scholes call option formula
        
        S = V * N(d1) - D * e^(-rT) * N(d2)
        where d2 = d1 - sigma_V * sqrt(T)
        """
        d1 = self._black_scholes_d1(V, sigma_V)
        d2 = d1 - sigma_V * np.sqrt(self.T)
        
        call_value = (
            V * norm.cdf(d1) - 
            self.D * np.exp(-self.r * self.T) * norm.cdf(d2)
        )
        return call_value
    
    def _equity_volatility_from_asset_vol(self, V, sigma_V):
        """
        Compute equity volatility given asset volatility
        
        Uses relationship: σ_S = (∂S/∂V) * (σ_V * V / S)
        
        where ∂S/∂V = N(d1) (delta of the call option)
        """
        d1 = self._black_scholes_d1(V, sigma_V)
        delta = norm.cdf(d1)  # ∂S/∂V
        
        # Equity volatility
        sigma_S_computed = delta * (sigma_V * V) / self.S
        return sigma_S_computed
    
    def objective_function(self, sigma_V):
        """
        Objective function to minimize: F(σ_V) = 0
        
        We want: σ_S_computed(σ_V) = σ_S_observed
        
        So: F(σ_V) = σ_S_computed(σ_V) - σ_S_observed
        
        Newton-Raphson will find σ_V where F(σ_V) = 0
        """
        # Asset value: V = S + PV(Debt)
        V = self.S + self.D * np.exp(-self.r * self.T)
        
        # Compute equity vol implied by this sigma_V
        sigma_S_computed = self._equity_volatility_from_asset_vol(V, sigma_V)
        
        # Objective: difference between computed and observed equity vol
        return sigma_S_computed - self.sigma_S
    
    def calibrate(self, max_iterations=100, tolerance=1e-6):
        """
        Calibrate using Newton-Raphson method
        
        Args:
            max_iterations (int): Maximum Newton-Raphson iterations
            tolerance (float): Convergence tolerance
        
        Returns:
            dict: {
                'asset_vol': σ_V (float),
                'converged': bool,
                'iterations': int,
                'final_error': float
            }
        """
        try:
            # Use SciPy's Newton-Raphson solver
            asset_vol = newton(
                self.objective_function,
                x0=self.initial_guess,
                maxiter=max_iterations,
                tol=tolerance
            )
            
            # Verify convergence
            final_error = abs(self.objective_function(asset_vol))
            converged = final_error < tolerance
            
            return {
                'asset_vol': asset_vol,
                'converged': converged,
                'final_error': final_error,
                'initial_guess': self.initial_guess
            }
        
        except RuntimeError as e:
            # Newton-Raphson failed to converge
            return {
                'asset_vol': None,
                'converged': False,
                'error': str(e),
                'initial_guess': self.initial_guess
            }


def calibrate_merton(S, equity_vol, D, T, r, max_iter=100):
    """
    Convenience function: Calibrate Merton model in one call
    
    Args:
        S (float): Stock price
        equity_vol (float): Equity volatility
        D (float): Total debt
        T (float): Time to maturity (years)
        r (float): Risk-free rate
        max_iter (int): Max iterations for Newton-Raphson
    
    Returns:
        float: Asset volatility (σ_V)
    
    Raises:
        ValueError: If calibration fails to converge
    """
    calibrator = MertonCalibration(S, equity_vol, D, T, r)
    result = calibrator.calibrate(max_iterations=max_iter)
    
    if not result['converged']:
        raise ValueError(
            f"Calibration failed to converge. "
            f"Error: {result.get('final_error', 'unknown')}. "
            f"Consider adjusting initial guess or parameters."
        )
    
    return result['asset_vol']


# ============================================================================
# TEST CASES
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MERTON CALIBRATION - TEST CASES")
    print("=" * 70)
    
    # Test 1: Simple synthetic example
    print("\nTest 1: Synthetic Company Data")
    print("-" * 70)
    
    S = 150.0           # Stock price
    equity_vol = 0.25   # Equity volatility (25%)
    D = 50.0            # Total debt
    T = 1.0             # 1 year to maturity
    r = 0.05            # 5% risk-free rate
    
    print(f"Stock Price (S):      ${S:.2f}")
    print(f"Equity Volatility:    {equity_vol:.2%}")
    print(f"Total Debt (D):       ${D:.2f}")
    print(f"Maturity (T):         {T:.1f} years")
    print(f"Risk-free rate (r):   {r:.2%}")
    
    calibrator = MertonCalibration(S, equity_vol, D, T, r)
    result = calibrator.calibrate()
    
    if result['converged']:
        asset_vol = result['asset_vol']
        print(f"\n✅ Calibration CONVERGED")
        print(f"Asset Volatility:     {asset_vol:.2%}")
        print(f"Calibration Error:    {result['final_error']:.2e}")
        print(f"Initial Guess:        {result['initial_guess']:.2%}")
    else:
        print(f"\n❌ Calibration FAILED")
        print(f"Error: {result.get('error', 'Unknown')}")
    
    # Test 2: Different leverage scenario
    print("\n" + "=" * 70)
    print("Test 2: High Leverage Scenario")
    print("-" * 70)
    
    S2 = 100.0
    equity_vol2 = 0.40  # Higher equity volatility (more risky)
    D2 = 100.0          # High debt (high leverage)
    
    print(f"Stock Price (S):      ${S2:.2f}")
    print(f"Equity Volatility:    {equity_vol2:.2%}")
    print(f"Total Debt (D):       ${D2:.2f}")
    print(f"Leverage (D/E):       {D2/S2:.2f}")
    
    calibrator2 = MertonCalibration(S2, equity_vol2, D2, T, r)
    result2 = calibrator2.calibrate()
    
    if result2['converged']:
        asset_vol2 = result2['asset_vol']
        print(f"\n✅ Calibration CONVERGED")
        print(f"Asset Volatility:     {asset_vol2:.2%}")
        print(f"Calibration Error:    {result2['final_error']:.2e}")
    else:
        print(f"\n❌ Calibration FAILED")
    
    # Test 3: Verify relationship
    print("\n" + "=" * 70)
    print("Test 3: Verify Calibration (Sanity Check)")
    print("-" * 70)
    print(f"Original equity vol:  {equity_vol:.2%}")
    print(f"Calibrated asset vol: {result['asset_vol']:.2%}")
    print(f"Asset vol < Equity vol? {result['asset_vol'] < equity_vol}")
    print("(Expected: YES, because equity is leveraged)")
    
    # Test 4: Edge case - very low debt
    print("\n" + "=" * 70)
    print("Test 4: Low Leverage (Low Debt)")
    print("-" * 70)
    
    S3 = 200.0
    equity_vol3 = 0.20
    D3 = 10.0  # Very low debt
    
    print(f"Stock Price (S):      ${S3:.2f}")
    print(f"Equity Volatility:    {equity_vol3:.2%}")
    print(f"Total Debt (D):       ${D3:.2f}")
    print(f"Leverage (D/E):       {D3/S3:.2f}")
    
    calibrator3 = MertonCalibration(S3, equity_vol3, D3, T, r)
    result3 = calibrator3.calibrate()
    
    if result3['converged']:
        asset_vol3 = result3['asset_vol']
        print(f"\n✅ Calibration CONVERGED")
        print(f"Asset Volatility:     {asset_vol3:.2%}")
        print(f"Asset vol ≈ Equity vol? {np.isclose(asset_vol3, equity_vol3, atol=0.01)}")
        print("(Expected: YES, because low leverage means asset ≈ equity)")
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)