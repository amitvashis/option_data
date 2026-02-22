"""
Compute Implied Volatility (IV) and Greeks for F&O option data
using the Black-Scholes model.

OPTIMIZED: Uses multiprocessing for parallel IV computation + vectorized Greeks.

Reads:  master/master_fo_data.csv
Writes: master/master_fo_data_with_greeks.csv

New columns: IV, Delta, Gamma, Vega, Theta, Rho, Moneyness
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
import multiprocessing as mp
import time
import warnings
import os

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────
RISK_FREE_RATE = 0.068
DIVIDEND_YIELD = 0.0
IV_LOWER = 0.001
IV_UPPER = 5.0

INPUT_FILE = "D:/antigravity/option_data/master/master_fo_data.csv"
OUTPUT_FILE = "D:/antigravity/option_data/master/master_fo_data_with_greeks.csv"

N_WORKERS = max(1, os.cpu_count() - 1)  # Leave 1 core free


# ── IV Solver (scalar, called per row in parallel) ───────────

def _bs_price(S, K, T, r, q, sigma, is_call):
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    if is_call:
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def _solve_iv(args):
    """Solve IV for one row. Designed for use with multiprocessing Pool."""
    S, K, T, r, q, mkt_price, is_call = args
    if T <= 0 or mkt_price <= 0 or S <= 0 or K <= 0:
        return np.nan
    try:
        f_lo = _bs_price(S, K, T, r, q, IV_LOWER, is_call) - mkt_price
        f_hi = _bs_price(S, K, T, r, q, IV_UPPER, is_call) - mkt_price
        if f_lo * f_hi > 0:
            return np.nan
        return brentq(lambda sig: _bs_price(S, K, T, r, q, sig, is_call) - mkt_price,
                      IV_LOWER, IV_UPPER, maxiter=100, xtol=1e-8)
    except (ValueError, RuntimeError):
        return np.nan


def _solve_iv_chunk(chunk_args):
    """Process a chunk of rows and return IV array. Used to reduce IPC overhead."""
    results = []
    for args in chunk_args:
        results.append(_solve_iv(args))
    return results


# ── Vectorized Greeks ────────────────────────────────────────

def compute_greeks_vectorized(S, K, T, r, q, sigma, is_call):
    valid = np.isfinite(sigma) & (T > 0) & (S > 0) & (K > 0) & (sigma > 0)
    delta = np.full_like(sigma, np.nan)
    gamma = np.full_like(sigma, np.nan)
    vega = np.full_like(sigma, np.nan)
    theta = np.full_like(sigma, np.nan)
    rho = np.full_like(sigma, np.nan)

    if not valid.any():
        return delta, gamma, vega, theta, rho

    Sv, Kv, Tv, sigv = S[valid], K[valid], T[valid], sigma[valid]
    is_call_v = is_call[valid]

    sqrt_T = np.sqrt(Tv)
    d1 = (np.log(Sv / Kv) + (r - q + 0.5 * sigv**2) * Tv) / (sigv * sqrt_T)
    d2 = d1 - sigv * sqrt_T

    exp_qT = np.exp(-q * Tv)
    exp_rT = np.exp(-r * Tv)
    phi_d1 = norm.pdf(d1)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    Nmd1 = norm.cdf(-d1)
    Nmd2 = norm.cdf(-d2)

    gamma[valid] = exp_qT * phi_d1 / (Sv * sigv * sqrt_T)
    vega[valid] = Sv * exp_qT * phi_d1 * sqrt_T / 100.0
    delta[valid] = np.where(is_call_v, exp_qT * Nd1, exp_qT * (Nd1 - 1))

    common_theta = -(Sv * phi_d1 * sigv * exp_qT) / (2 * sqrt_T)
    theta_call = common_theta - r * Kv * exp_rT * Nd2 + q * Sv * exp_qT * Nd1
    theta_put = common_theta + r * Kv * exp_rT * Nmd2 - q * Sv * exp_qT * Nmd1
    theta[valid] = np.where(is_call_v, theta_call, theta_put) / 365.0

    rho_call = Kv * Tv * exp_rT * Nd2 / 100.0
    rho_put = -Kv * Tv * exp_rT * Nmd2 / 100.0
    rho[valid] = np.where(is_call_v, rho_call, rho_put)

    return delta, gamma, vega, theta, rho


# ── Main ─────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Black-Scholes IV & Greeks (Multiprocessing)")
    print(f"  Workers: {N_WORKERS}")
    print("=" * 60)

    r, q = RISK_FREE_RATE, DIVIDEND_YIELD

    # Load data
    print(f"\nLoading: {INPUT_FILE}")
    t0 = time.time()
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    # Prepare
    print("Preparing data...")
    trade_dt = pd.to_datetime(df["TradDt"])
    expiry_dt = pd.to_datetime(df["XpryDt"])
    T_arr = ((expiry_dt - trade_dt).dt.days / 365.0).values.astype(np.float64)
    mkt_price = df["SttlmPric"].fillna(df["ClsPric"]).values.astype(np.float64)
    S = df["UndrlygPric"].values.astype(np.float64)
    K = df["StrkPric"].values.astype(np.float64)
    is_call = (df["OptnTp"] == "CE").values

    df["Moneyness"] = S / K

    # ── Build argument tuples ────────────────────────────────
    print("Building IV solver arguments...")
    all_args = list(zip(S, K, T_arr, [r]*len(df), [q]*len(df), mkt_price, is_call))

    # Split into chunks for workers (reduce IPC overhead)
    SUB_CHUNK = 5000  # rows per sub-task sent to a worker
    chunks = [all_args[i:i+SUB_CHUNK] for i in range(0, len(all_args), SUB_CHUNK)]
    print(f"  {len(chunks)} chunks of ~{SUB_CHUNK} rows each")

    # ── Compute IV in parallel ───────────────────────────────
    print(f"\nComputing IV with {N_WORKERS} workers...")
    t_iv = time.time()

    iv_arr = np.full(len(df), np.nan)
    done = 0

    with mp.Pool(N_WORKERS) as pool:
        for i, result in enumerate(pool.imap(_solve_iv_chunk, chunks)):
            start_idx = i * SUB_CHUNK
            end_idx = start_idx + len(result)
            iv_arr[start_idx:end_idx] = result
            done += len(result)

            # Progress every 10 chunks
            if (i + 1) % 10 == 0 or done == len(df):
                elapsed = time.time() - t_iv
                speed = done / elapsed if elapsed > 0 else 0
                eta = (len(df) - done) / speed if speed > 0 else 0
                print(f"  {done:>10,} / {len(df):,} "
                      f"| {elapsed:>6.1f}s "
                      f"| {speed:>8,.0f} rows/s "
                      f"| ETA: {eta/60:>5.1f} min")

    df["IV"] = iv_arr
    iv_time = time.time() - t_iv
    print(f"\nIV done in {iv_time:.1f}s ({iv_time/60:.1f} min)")
    print(f"  Valid IVs: {np.isfinite(iv_arr).sum():,} / {len(df):,}")

    # ── Greeks (vectorized, near-instant) ────────────────────
    print("\nComputing Greeks (vectorized)...")
    t_g = time.time()
    delta, gamma, vega, theta, rho = compute_greeks_vectorized(S, K, T_arr, r, q, iv_arr, is_call)
    df["Delta"] = delta
    df["Gamma"] = gamma
    df["Vega"] = vega
    df["Theta"] = theta
    df["Rho"] = rho
    print(f"Greeks computed in {time.time()-t_g:.1f}s")

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    greek_cols = ["IV", "Delta", "Gamma", "Vega", "Theta", "Rho", "Moneyness"]
    print(f"\n{'Col':<12} {'Valid':>10} {'NaN':>10} {'Mean':>12} {'Median':>12} {'Min':>12} {'Max':>12}")
    print("-" * 82)
    for col in greek_cols:
        v = df[col].dropna()
        n = df[col].isna().sum()
        if len(v) > 0:
            print(f"{col:<12} {len(v):>10,} {n:>10,} {v.mean():>12.6f} {v.median():>12.6f} {v.min():>12.6f} {v.max():>12.6f}")

    # Sanity
    print("\n  Sanity Checks:")
    cd = df[df["OptnTp"]=="CE"]["Delta"].dropna()
    pd_ = df[df["OptnTp"]=="PE"]["Delta"].dropna()
    print(f"    Call Delta: [{cd.min():.4f}, {cd.max():.4f}] (expect ~[0,1])")
    print(f"    Put Delta:  [{pd_.min():.4f}, {pd_.max():.4f}] (expect ~[-1,0])")
    print(f"    Gamma >= 0: {(df['Gamma'].dropna() >= -1e-10).all()}")
    print(f"    Vega >= 0:  {(df['Vega'].dropna() >= -1e-10).all()}")

    # Save
    print(f"\nSaving to: {OUTPUT_FILE}")
    t_s = time.time()
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved in {time.time()-t_s:.1f}s")
    total = time.time() - t0
    print(f"\nTotal time: {total/60:.1f} minutes")
    print(f"Final shape: {df.shape}")
    print("\nDone!")


if __name__ == "__main__":
    main()
