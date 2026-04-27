"""
visualize_greeks.py
===================
Generate Black-Scholes Greek visualizations.

Produces two sets of figures:
1. Greek surfaces over the (spot, time-to-expiry) plane for calls and puts.
   These show how delta, gamma, vega, theta, and rho behave across the
   moneyness/maturity space.
2. Analytical-vs-numerical validation plots for a fixed time slice. The two
   curves should be visually indistinguishable, which validates the
   analytical Greek formulas against finite differences.

All figures are saved to figures/ as PNGs.
"""

from __future__ import annotations
import os

import matplotlib.pyplot as plt
import numpy as np

from black_scholes import greeks, numerical_greeks


FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Greek surfaces over (spot, time)
# ---------------------------------------------------------------------------

def compute_greek_grids(
    K: float = 100.0,
    r: float = 0.05,
    sigma: float = 0.20,
    option_type: str = "call",
    spot_range: tuple = (60.0, 140.0),
    time_range: tuple = (0.05, 1.0),
    n_grid: int = 60,
):
    """Compute Greeks on a (spot, time) grid for plotting."""
    spots = np.linspace(*spot_range, n_grid)
    times = np.linspace(*time_range, n_grid)
    S_grid, T_grid = np.meshgrid(spots, times)

    arrays = {name: np.zeros_like(S_grid) for name in
              ("delta", "gamma", "vega", "theta", "rho")}

    for i in range(n_grid):
        for j in range(n_grid):
            g = greeks(S_grid[i, j], K, T_grid[i, j], r, sigma, option_type)
            for name in arrays:
                arrays[name][i, j] = getattr(g, name)

    return S_grid, T_grid, arrays


def plot_greek_surfaces(option_type: str = "call", K: float = 100.0):
    """2x3 grid of heatmaps for delta, gamma, vega, theta, rho."""
    S_grid, T_grid, g = compute_greek_grids(K=K, option_type=option_type)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Black-Scholes Greek Surfaces: European {option_type.capitalize()} "
        f"(K={K:.0f}, r=5%, sigma=20%)",
        fontsize=14, fontweight="bold"
    )

    panels = [
        ("delta", "RdBu_r", "Delta (dV/dS)"),
        ("gamma", "viridis", "Gamma (d2V/dS2)"),
        ("vega", "magma", "Vega (dV/dsigma)"),
        ("theta", "RdBu_r", "Theta (dV/dt, calendar)"),
        ("rho", "RdBu_r", "Rho (dV/dr)"),
    ]

    for idx, (name, cmap, title) in enumerate(panels):
        ax = axes[idx // 3, idx % 3]
        c = ax.pcolormesh(S_grid, T_grid, g[name], shading="auto", cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Spot price")
        ax.set_ylabel("Time to expiry (years)")
        ax.axvline(K, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
        plt.colorbar(c, ax=ax, fraction=0.046, pad=0.04)

    # Use the unused 6th subplot for an annotation panel
    ax6 = axes[1, 2]
    ax6.axis("off")
    annotation = (
        "Reading the surfaces:\n\n"
        "• Delta: 0 to 1 for calls; sharpest near\n"
        "  the strike close to expiry.\n\n"
        "• Gamma: peaks at-the-money and\n"
        "  diverges as T -> 0.\n\n"
        "• Vega: largest for ATM options with\n"
        "  long maturities.\n\n"
        "• Theta: most negative (fastest decay)\n"
        "  for ATM options near expiry.\n\n"
        "• Rho: scales with K * T * exp(-rT) * N(d2);\n"
        "  matters most for long-dated options.\n\n"
        "Dashed line in each panel marks the\n"
        f"strike (K = {K:.0f})."
    )
    ax6.text(0.05, 0.95, annotation, transform=ax6.transAxes, fontsize=9,
             verticalalignment="top", family="monospace")

    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"greeks_{option_type}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Analytical vs numerical validation
# ---------------------------------------------------------------------------

def plot_validation(
    K: float = 100.0,
    T: float = 0.5,
    r: float = 0.05,
    sigma: float = 0.20,
    option_type: str = "call",
):
    """
    Overlay analytical and numerical Greeks across a range of spot prices,
    holding T fixed. Curves should be visually indistinguishable.
    """
    spots = np.linspace(60, 140, 50)
    g_analytical = [greeks(s, K, T, r, sigma, option_type) for s in spots]
    g_numerical = [numerical_greeks(s, K, T, r, sigma, option_type) for s in spots]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        f"Analytical vs Numerical Greeks ({option_type}, K={K:.0f}, "
        f"T={T}, r=5%, sigma=20%)",
        fontsize=13, fontweight="bold"
    )

    panels = ["delta", "gamma", "vega", "theta", "rho"]

    for idx, name in enumerate(panels):
        ax = axes[idx // 3, idx % 3]
        analytical = np.array([getattr(g, name) for g in g_analytical])
        numerical = np.array([getattr(g, name) for g in g_numerical])
        ax.plot(spots, analytical, "-", label="Analytical", linewidth=2.2)
        ax.plot(spots, numerical, "--", label="Numerical (FD)", linewidth=1.4)
        ax.axvline(K, color="gray", linestyle=":", alpha=0.6)
        ax.set_title(name.capitalize(), fontsize=11)
        ax.set_xlabel("Spot")
        ax.set_ylabel(name)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    # Sixth panel: log of absolute error
    ax6 = axes[1, 2]
    for name in panels:
        a = np.array([getattr(g, name) for g in g_analytical])
        n = np.array([getattr(g, name) for g in g_numerical])
        err = np.abs(a - n) + 1e-16
        ax6.semilogy(spots, err, label=name)
    ax6.set_title("|Analytical - Numerical|, log scale", fontsize=11)
    ax6.set_xlabel("Spot")
    ax6.set_ylabel("Absolute error")
    ax6.legend(fontsize=8, loc="best")
    ax6.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"validation_{option_type}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Volatility smile-style figure (optional context plot)
# ---------------------------------------------------------------------------

def plot_price_curves(K: float = 100.0, r: float = 0.05, sigma: float = 0.20):
    """Plot call and put prices across moneyness for several maturities."""
    from black_scholes import price as bs_price

    spots = np.linspace(60, 140, 100)
    maturities = [0.083, 0.25, 0.5, 1.0]  # 1m, 3m, 6m, 1y

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for T in maturities:
        calls = [bs_price(s, K, T, r, sigma, "call") for s in spots]
        puts = [bs_price(s, K, T, r, sigma, "put") for s in spots]
        ax1.plot(spots, calls, label=f"T={T:.2f}y")
        ax2.plot(spots, puts, label=f"T={T:.2f}y")

    for ax, kind in zip((ax1, ax2), ("Call", "Put")):
        ax.axvline(K, color="gray", linestyle="--", alpha=0.5, label=f"K={K:.0f}")
        ax.plot(spots,
                np.maximum(spots - K, 0) if kind == "Call" else np.maximum(K - spots, 0),
                color="black", linestyle=":", alpha=0.6, label="Intrinsic")
        ax.set_title(f"{kind} price vs spot", fontsize=12)
        ax.set_xlabel("Spot price")
        ax.set_ylabel("Option price")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"European Option Prices (K={K:.0f}, r=5%, sigma=20%)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "price_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    print("Generating figures...")
    plot_greek_surfaces(option_type="call")
    plot_greek_surfaces(option_type="put")
    plot_validation(option_type="call")
    plot_validation(option_type="put")
    plot_price_curves()
    print("Done.")
