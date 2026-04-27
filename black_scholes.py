"""
black_scholes.py
================
Black-Scholes-Merton option pricer with analytical and numerical Greeks.

Closed-form pricing for European options on stocks with continuous dividend
yield, the five primary Greeks computed analytically, and finite-difference
Greeks for validation. Includes a Newton-Raphson implied volatility solver.

References
----------
- Hull, J. C. (2018). Options, Futures, and Other Derivatives, 10th ed.
- Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate
  Liabilities. Journal of Political Economy, 81(3), 637-654.
- Merton, R. C. (1973). Theory of Rational Option Pricing. Bell Journal of
  Economics and Management Science, 4(1), 141-183.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import norm


OptionType = Literal["call", "put"]


# ---------------------------------------------------------------------------
# Internal helpers: d1, d2
# ---------------------------------------------------------------------------

def _d1(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    The d1 term from the Black-Scholes formula.

        d1 = [ln(S/K) + (r - q + 0.5 * sigma^2) * T] / (sigma * sqrt(T))
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if S <= 0 or K <= 0:
        raise ValueError(f"S and K must be positive, got S={S}, K={K}")
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """The d2 term: d2 = d1 - sigma * sqrt(T)."""
    return _d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

def price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    q: float = 0.0,
) -> float:
    """
    Black-Scholes price for a European option.

    Parameters
    ----------
    S : float       Spot price of the underlying.
    K : float       Strike price.
    T : float       Time to expiry, in years (e.g., 0.25 for three months).
    r : float       Risk-free rate, continuously compounded (e.g., 0.05 for 5%).
    sigma : float   Volatility of the underlying, annualized (e.g., 0.20 for 20%).
    option_type : "call" or "put".
    q : float       Continuous dividend yield, default 0.0.

    Returns
    -------
    float           Option price in the same units as S.
    """
    d1 = _d1(S, K, T, r, sigma, q)
    d2 = _d2(S, K, T, r, sigma, q)

    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


# ---------------------------------------------------------------------------
# Analytical Greeks
# ---------------------------------------------------------------------------

@dataclass
class Greeks:
    """Container for the five primary Greeks."""
    delta: float   # dV/dS
    gamma: float   # d^2V/dS^2  (same for call and put)
    vega: float    # dV/dsigma  (per 1.00 vol move; divide by 100 for per-1%)
    theta: float   # dV/dT      (per year; divide by 365 for per-day)
    rho: float     # dV/dr      (per 1.00 rate move; divide by 100 for per-1%)

    def per_pct(self) -> dict:
        """Return Greeks rescaled to per-percentage-point conventions."""
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "vega_per_1pct_vol": self.vega / 100,
            "theta_per_day": self.theta / 365,
            "rho_per_1pct_rate": self.rho / 100,
        }


def greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    q: float = 0.0,
) -> Greeks:
    """
    Closed-form Greeks for a European option under Black-Scholes-Merton.

    Formulas (with continuous dividend yield q):
        delta_call =  exp(-q*T) * N(d1)
        delta_put  = -exp(-q*T) * N(-d1)
        gamma      =  exp(-q*T) * n(d1) / (S * sigma * sqrt(T))
        vega       =  S * exp(-q*T) * n(d1) * sqrt(T)
        theta_call = -S * exp(-q*T) * n(d1) * sigma / (2*sqrt(T))
                     - r * K * exp(-r*T) * N(d2)
                     + q * S * exp(-q*T) * N(d1)
        theta_put  = -S * exp(-q*T) * n(d1) * sigma / (2*sqrt(T))
                     + r * K * exp(-r*T) * N(-d2)
                     - q * S * exp(-q*T) * N(-d1)
        rho_call   =  K * T * exp(-r*T) * N(d2)
        rho_put    = -K * T * exp(-r*T) * N(-d2)

    where n() is the standard normal PDF and N() is the CDF.
    """
    d1 = _d1(S, K, T, r, sigma, q)
    d2 = _d2(S, K, T, r, sigma, q)
    pdf_d1 = norm.pdf(d1)

    # gamma and vega are the same for calls and puts
    gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T)

    if option_type == "call":
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (
            -S * np.exp(-q * T) * pdf_d1 * sigma / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + q * S * np.exp(-q * T) * norm.cdf(d1)
        )
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (
            -S * np.exp(-q * T) * pdf_d1 * sigma / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1)
        )
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


# ---------------------------------------------------------------------------
# Numerical (finite-difference) Greeks for validation
# ---------------------------------------------------------------------------

def numerical_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    q: float = 0.0,
    h_S: float = 0.01,
    h_sigma: float = 1e-4,
    h_T: float = 1e-4,
    h_r: float = 1e-4,
) -> Greeks:
    """
    Greeks computed by central finite differences against the price function.

    Used as a validation check: numerical and analytical Greeks should agree
    to within step-size truncation error (~1e-4 with the defaults).

    delta:  [V(S+h) - V(S-h)] / (2h)
    gamma:  [V(S+h) - 2*V(S) + V(S-h)] / h^2
    vega:   [V(sigma+h) - V(sigma-h)] / (2h)
    theta:  -[V(T+h) - V(T-h)] / (2h)    (note sign: theta is dV/dT but
                                          conventionally reported as decay,
                                          and we follow the analytical formula
                                          which uses dV/dT directly)
    rho:    [V(r+h) - V(r-h)] / (2h)
    """
    def p(S_=S, K_=K, T_=T, r_=r, sigma_=sigma, q_=q):
        return price(S_, K_, T_, r_, sigma_, option_type, q_)

    p_base = p()

    delta = (p(S_=S + h_S) - p(S_=S - h_S)) / (2 * h_S)
    gamma = (p(S_=S + h_S) - 2 * p_base + p(S_=S - h_S)) / (h_S ** 2)
    vega = (p(sigma_=sigma + h_sigma) - p(sigma_=sigma - h_sigma)) / (2 * h_sigma)
    # Theta uses Hull's convention: dV/dt (calendar time), which equals -dV/dT
    # (time to expiry). So we negate the central difference in T.
    theta = -(p(T_=T + h_T) - p(T_=T - h_T)) / (2 * h_T)
    rho = (p(r_=r + h_r) - p(r_=r - h_r)) / (2 * h_r)

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


# ---------------------------------------------------------------------------
# Implied volatility (Newton-Raphson)
# ---------------------------------------------------------------------------

def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType = "call",
    q: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 100,
    sigma_init: float = 0.2,
) -> float:
    """
    Solve for the volatility that reproduces a given market price using
    Newton-Raphson with vega as the derivative.

    Newton-Raphson update:
        sigma_{n+1} = sigma_n - [price(sigma_n) - market_price] / vega(sigma_n)

    Returns NaN if the solver does not converge or vega becomes too small
    (which happens for deep ITM/OTM options where the price is insensitive
    to volatility).
    """
    # Sanity check: market price must be inside the no-arbitrage bounds.
    intrinsic_call = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    intrinsic_put = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
    intrinsic = intrinsic_call if option_type == "call" else intrinsic_put
    upper = S * np.exp(-q * T) if option_type == "call" else K * np.exp(-r * T)
    if not (intrinsic <= market_price <= upper + 1e-10):
        return float("nan")

    sigma = sigma_init
    for _ in range(max_iter):
        bs_price = price(S, K, T, r, sigma, option_type, q)
        diff = bs_price - market_price
        if abs(diff) < tol:
            return sigma
        v = greeks(S, K, T, r, sigma, option_type, q).vega
        if abs(v) < 1e-10:
            return float("nan")
        sigma = sigma - diff / v
        if sigma <= 0:
            # Reset to a small positive value if Newton overshoots.
            sigma = 1e-4
    return float("nan")


# ---------------------------------------------------------------------------
# Self-check against Hull textbook reference values
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Hull, Example 15.6 (10th ed.):
    # S=42, K=40, T=0.5, r=0.10, sigma=0.20
    # Expected call price ~4.76, put price ~0.81
    S, K, T, r, sigma = 42.0, 40.0, 0.5, 0.10, 0.20

    call = price(S, K, T, r, sigma, "call")
    put = price(S, K, T, r, sigma, "put")

    print("Black-Scholes Self-Check")
    print("=" * 50)
    print(f"Inputs: S={S}, K={K}, T={T}, r={r}, sigma={sigma}")
    print(f"\nCall price: {call:.4f}  (Hull expected ~4.76)")
    print(f"Put price:  {put:.4f}  (Hull expected ~0.81)")

    parity_lhs = call - put
    parity_rhs = S - K * np.exp(-r * T)
    print(f"\nPut-call parity: C - P = {parity_lhs:.6f}")
    print(f"                 S - Ke^-rT = {parity_rhs:.6f}")
    print(f"                 |diff| = {abs(parity_lhs - parity_rhs):.2e}")

    g = greeks(S, K, T, r, sigma, "call")
    ng = numerical_greeks(S, K, T, r, sigma, "call")
    print(f"\nCall Greeks at S={S}:")
    print(f"  Greek      Analytical       Numerical        Diff")
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        a = getattr(g, name)
        n = getattr(ng, name)
        print(f"  {name:<8}  {a:14.6f}  {n:14.6f}  {abs(a - n):.2e}")

    # Implied vol round-trip: price an option, recover its sigma.
    market = price(100, 100, 1.0, 0.05, 0.25, "call")
    iv = implied_volatility(market, 100, 100, 1.0, 0.05, "call")
    print(f"\nImplied vol round-trip:")
    print(f"  Priced call at sigma=0.2500, market price = {market:.4f}")
    print(f"  Recovered IV from price: {iv:.6f}")
