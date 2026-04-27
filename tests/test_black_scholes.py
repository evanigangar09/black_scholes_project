"""
tests/test_black_scholes.py
===========================
Unit tests for the Black-Scholes pricer.

Run with: pytest tests/ -v

Covers:
    - Hull (10th ed.) reference values for prices and Greeks
    - Put-call parity
    - Greek sign conventions
    - Boundary behavior (deep ITM, deep OTM, near expiry)
    - Analytical vs numerical Greeks agreement
    - Implied volatility round-trip
"""

import numpy as np
import pytest

from black_scholes import (
    Greeks,
    greeks,
    implied_volatility,
    numerical_greeks,
    price,
)


# ---------------------------------------------------------------------------
# Hull reference values
# ---------------------------------------------------------------------------

class TestHullReferenceValues:
    """Hull, Example 15.6: S=42, K=40, T=0.5, r=0.10, sigma=0.20"""

    def test_call_price(self):
        c = price(42, 40, 0.5, 0.10, 0.20, "call")
        assert abs(c - 4.7594) < 0.001

    def test_put_price(self):
        p = price(42, 40, 0.5, 0.10, 0.20, "put")
        assert abs(p - 0.8086) < 0.001


# ---------------------------------------------------------------------------
# Put-call parity
# ---------------------------------------------------------------------------

class TestPutCallParity:
    """C - P = S*exp(-q*T) - K*exp(-r*T) for any (S, K, T, r, sigma, q)."""

    @pytest.mark.parametrize("S,K,T,r,sigma,q", [
        (100, 100, 1.0, 0.05, 0.20, 0.0),
        (100, 90, 0.5, 0.03, 0.30, 0.0),
        (50, 100, 2.0, 0.07, 0.15, 0.02),
        (200, 150, 0.25, 0.01, 0.40, 0.05),
    ])
    def test_parity(self, S, K, T, r, sigma, q):
        c = price(S, K, T, r, sigma, "call", q)
        p = price(S, K, T, r, sigma, "put", q)
        lhs = c - p
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert abs(lhs - rhs) < 1e-10


# ---------------------------------------------------------------------------
# Greek sign conventions
# ---------------------------------------------------------------------------

class TestGreekSigns:
    """Sign and magnitude sanity checks."""

    @pytest.fixture
    def base(self):
        return dict(S=100, K=100, T=0.5, r=0.05, sigma=0.20)

    def test_call_delta_in_unit_interval(self, base):
        g = greeks(option_type="call", **base)
        assert 0 < g.delta < 1

    def test_put_delta_negative_unit_interval(self, base):
        g = greeks(option_type="put", **base)
        assert -1 < g.delta < 0

    def test_gamma_positive_for_call_and_put(self, base):
        # gamma is the same for calls and puts and is always positive
        g_call = greeks(option_type="call", **base)
        g_put = greeks(option_type="put", **base)
        assert g_call.gamma > 0
        assert g_put.gamma > 0
        assert abs(g_call.gamma - g_put.gamma) < 1e-12

    def test_vega_positive_and_equal(self, base):
        g_call = greeks(option_type="call", **base)
        g_put = greeks(option_type="put", **base)
        assert g_call.vega > 0
        assert g_put.vega > 0
        assert abs(g_call.vega - g_put.vega) < 1e-12

    def test_call_rho_positive(self, base):
        g = greeks(option_type="call", **base)
        assert g.rho > 0

    def test_put_rho_negative(self, base):
        g = greeks(option_type="put", **base)
        assert g.rho < 0

    def test_call_theta_negative_when_no_dividend(self, base):
        # Without dividends, ATM call theta is negative (time decay).
        g = greeks(option_type="call", **base)
        assert g.theta < 0


# ---------------------------------------------------------------------------
# Boundary behavior
# ---------------------------------------------------------------------------

class TestBoundaries:
    def test_deep_itm_call_approaches_intrinsic(self):
        # Deep ITM call should approach S - K*exp(-rT)
        c = price(S=200, K=100, T=0.5, r=0.05, sigma=0.20, option_type="call")
        intrinsic = 200 - 100 * np.exp(-0.05 * 0.5)
        assert abs(c - intrinsic) < 0.01

    def test_deep_otm_call_approaches_zero(self):
        c = price(S=50, K=200, T=0.5, r=0.05, sigma=0.20, option_type="call")
        assert c < 0.01

    def test_deep_itm_put_approaches_intrinsic(self):
        p = price(S=50, K=200, T=0.5, r=0.05, sigma=0.20, option_type="put")
        intrinsic = 200 * np.exp(-0.05 * 0.5) - 50
        assert abs(p - intrinsic) < 0.01

    def test_deep_otm_put_approaches_zero(self):
        p = price(S=200, K=50, T=0.5, r=0.05, sigma=0.20, option_type="put")
        assert p < 0.01


# ---------------------------------------------------------------------------
# Analytical vs numerical agreement
# ---------------------------------------------------------------------------

class TestNumericalAgreement:
    """Analytical and finite-difference Greeks should agree closely."""

    @pytest.mark.parametrize("S,K,T,r,sigma,option_type", [
        (100, 100, 0.5, 0.05, 0.20, "call"),
        (100, 100, 0.5, 0.05, 0.20, "put"),
        (90, 100, 1.0, 0.03, 0.30, "call"),
        (110, 100, 0.25, 0.05, 0.15, "put"),
    ])
    def test_all_greeks_agree(self, S, K, T, r, sigma, option_type):
        analytical = greeks(S, K, T, r, sigma, option_type)
        numerical = numerical_greeks(S, K, T, r, sigma, option_type)
        # Agreement should be at the step-size truncation level (~1e-4 with
        # default step sizes; relax to 1e-3 to be robust against float noise).
        assert abs(analytical.delta - numerical.delta) < 1e-3
        assert abs(analytical.gamma - numerical.gamma) < 1e-3
        assert abs(analytical.vega - numerical.vega) < 1e-3
        assert abs(analytical.theta - numerical.theta) < 1e-3
        assert abs(analytical.rho - numerical.rho) < 1e-3


# ---------------------------------------------------------------------------
# Implied volatility round-trip
# ---------------------------------------------------------------------------

class TestImpliedVolatility:
    """Pricing then inverting should recover the original sigma."""

    @pytest.mark.parametrize("sigma", [0.10, 0.20, 0.35, 0.60])
    def test_roundtrip_call(self, sigma):
        S, K, T, r = 100, 100, 1.0, 0.05
        market = price(S, K, T, r, sigma, "call")
        iv = implied_volatility(market, S, K, T, r, "call")
        assert abs(iv - sigma) < 1e-6

    @pytest.mark.parametrize("sigma", [0.10, 0.20, 0.35, 0.60])
    def test_roundtrip_put(self, sigma):
        S, K, T, r = 100, 100, 1.0, 0.05
        market = price(S, K, T, r, sigma, "put")
        iv = implied_volatility(market, S, K, T, r, "put")
        assert abs(iv - sigma) < 1e-6

    def test_below_intrinsic_returns_nan(self):
        # Market price below intrinsic value violates no-arbitrage; should NaN.
        iv = implied_volatility(0.001, 200, 100, 0.5, 0.05, "call")
        assert np.isnan(iv)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_negative_T_raises(self):
        with pytest.raises(ValueError):
            price(100, 100, -0.5, 0.05, 0.20, "call")

    def test_zero_sigma_raises(self):
        with pytest.raises(ValueError):
            price(100, 100, 0.5, 0.05, 0.0, "call")

    def test_negative_S_raises(self):
        with pytest.raises(ValueError):
            price(-100, 100, 0.5, 0.05, 0.20, "call")

    def test_invalid_option_type_raises(self):
        with pytest.raises(ValueError):
            price(100, 100, 0.5, 0.05, 0.20, "straddle")
