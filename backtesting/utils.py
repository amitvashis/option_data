"""
Shared utilities for option strategy backtesting.
"""

import numpy as np
import pandas as pd
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

DATA_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "VI_data_extraction", "vi_data_historical.csv"
)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load_data(filepath=DATA_FILE):
    """Load and preprocess the VI data."""
    df = pd.read_csv(filepath)
    df["TradDt"] = pd.to_datetime(df["TradDt"])
    df["XpryDt"] = pd.to_datetime(df["XpryDt"])

    # Keep only option rows (CE/PE) with valid VI
    df = df[df["OptnTp"].isin(["CE", "PE"])].copy()
    df = df[df["VI"].notna() & (df["VI"] > 0)].copy()
    df = df[df["T"] > 0].copy()
    df = df[df["ClsPric"] > 0].copy()
    df = df[df["UndrlygPric"] > 0].copy()
    df = df[df["StrkPric"] > 0].copy()

    df.sort_values(["TradDt", "TckrSymb", "OptnTp", "XpryDt", "StrkPric"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def get_atm_options(df):
    """
    For each (TradDt, TckrSymb, OptnTp, XpryDt), pick the row
    where StrkPric is closest to UndrlygPric.
    """
    df = df.copy()
    df["_dist"] = abs(df["StrkPric"] - df["UndrlygPric"])
    idx = df.groupby(["TradDt", "TckrSymb", "OptnTp", "XpryDt"])["_dist"].idxmin()
    atm = df.loc[idx].drop(columns=["_dist"]).copy()
    atm.reset_index(drop=True, inplace=True)
    return atm


def compute_forward_factor(v1, t1, v2, t2):
    """
    Compute the Forward Factor from two IV / time-to-expiry pairs.

    Forward Variance = (V2^2 * T2 - V1^2 * T1) / (T2 - T1)
    Forward Volatility = sqrt(Forward Variance)
    Forward Factor = (V1 / Forward Volatility) - 1

    Parameters:
        v1: Front-month implied volatility (annualized)
        t1: Front-month time to expiry (years)
        v2: Back-month implied volatility (annualized)
        t2: Back-month time to expiry (years)

    Returns:
        forward_factor, forward_vol (numpy arrays or scalars)
    """
    v1 = np.asarray(v1, dtype=float)
    t1 = np.asarray(t1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    t2 = np.asarray(t2, dtype=float)

    dt = t2 - t1
    # Guard against zero or negative dt
    valid = dt > 1e-6

    forward_var = np.where(valid, (v2**2 * t2 - v1**2 * t1) / dt, np.nan)

    # Forward variance can be negative if term structure is inverted
    # In that case, forward vol is undefined
    fv_positive = forward_var > 0
    forward_vol = np.where(fv_positive, np.sqrt(forward_var), np.nan)

    # Forward Factor
    ff = np.where(np.isfinite(forward_vol) & (forward_vol > 1e-8),
                  (v1 / forward_vol) - 1.0,
                  np.nan)

    return ff, forward_vol


def calc_metrics(pnl_series, capital=100000):
    """
    Calculate performance metrics from a P&L series.

    Parameters:
        pnl_series: Series of individual trade P&Ls
        capital: Starting capital for CAGR calculation

    Returns:
        dict of metrics
    """
    pnl = np.array(pnl_series, dtype=float)
    pnl = pnl[np.isfinite(pnl)]

    if len(pnl) == 0:
        return {
            "total_trades": 0,
            "total_pnl": 0,
            "avg_pnl": 0,
            "win_rate": 0,
            "max_drawdown": 0,
            "sharpe": 0,
            "profit_factor": 0,
        }

    total_pnl = np.sum(pnl)
    avg_pnl = np.mean(pnl)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    win_rate = len(wins) / len(pnl) * 100 if len(pnl) > 0 else 0

    # Sharpe (annualized, assuming ~252 trades/year as rough proxy)
    if np.std(pnl) > 0:
        sharpe = (np.mean(pnl) / np.std(pnl)) * np.sqrt(min(252, len(pnl)))
    else:
        sharpe = 0

    # Profit Factor
    gross_profit = np.sum(wins) if len(wins) > 0 else 0
    gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 1e-8
    profit_factor = gross_profit / gross_loss

    # Max Drawdown from cumulative equity
    equity = np.cumsum(pnl) + capital
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_dd = np.max(drawdown) * 100

    return {
        "total_trades": len(pnl),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(avg_pnl, 2),
        "win_rate": round(win_rate, 1),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "profit_factor": round(profit_factor, 2),
    }


def print_metrics(metrics, strategy_name="Strategy"):
    """Pretty-print performance metrics."""
    print(f"\n{'='*60}")
    print(f"  {strategy_name} — Performance Summary")
    print(f"{'='*60}")
    for key, val in metrics.items():
        label = key.replace("_", " ").title()
        if "pct" in key.lower():
            print(f"  {label}: {val}%")
        elif "rate" in key.lower():
            print(f"  {label}: {val}%")
        else:
            print(f"  {label}: {val}")
    print(f"{'='*60}")


def ensure_results_dir():
    """Create results directory if it doesn't exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR
