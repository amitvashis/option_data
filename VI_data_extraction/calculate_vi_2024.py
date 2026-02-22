"""
Calculate Implied Volatility (IV) for 2024 F&O data from master_fo_data.csv
Uses VECTORIZED Newton-Raphson Black-Scholes IV solver for speed.
Saves result as vi_data_historical.csv in VI_data_extraction folder.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm

# ── Configuration ───────────────────────────────────────────────────────────

INPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "master", "master_fo_data.csv")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "vi_data_historical.csv")

RISK_FREE_RATE = 0.065  # ~6.5% India 10yr bond yield


# ── Vectorized IV Solver ────────────────────────────────────────────────────

def vectorized_iv(market_price, S, K, T, r, is_call, tol=1e-6, max_iter=100):
    """
    Vectorized Newton-Raphson implied volatility solver.
    All inputs are numpy arrays of the same length.
    Returns numpy array of IV values (NaN where it fails).
    """
    n = len(market_price)
    sigma = np.full(n, 0.3)  # initial guess
    result = np.full(n, np.nan)

    # Mask for valid rows that are still being iterated
    active = np.ones(n, dtype=bool)

    for iteration in range(max_iter):
        if not active.any():
            break

        s = sigma[active]
        mp = market_price[active]
        s_price = S[active]
        k = K[active]
        t = T[active]
        call = is_call[active]

        sqrt_t = np.sqrt(t)
        d1 = (np.log(s_price / k) + (r + 0.5 * s ** 2) * t) / (s * sqrt_t)
        d2 = d1 - s * sqrt_t

        # Black-Scholes price
        call_price = s_price * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
        put_price = k * np.exp(-r * t) * norm.cdf(-d2) - s_price * norm.cdf(-d1)
        bs_price = np.where(call, call_price, put_price)

        # Vega
        vega = s_price * norm.pdf(d1) * sqrt_t

        diff = bs_price - mp

        # Converged
        converged = np.abs(diff) < tol
        # Bad vega
        bad_vega = vega < 1e-12

        # Get indices in the full array
        active_indices = np.where(active)[0]

        # Mark converged
        converged_idx = active_indices[converged]
        result[converged_idx] = sigma[converged_idx]

        # Mark failed (bad vega but not converged)
        failed_idx = active_indices[bad_vega & ~converged]
        # result stays NaN for failed

        # Update sigma for remaining
        still_going = ~converged & ~bad_vega
        update_idx = active_indices[still_going]
        sigma[update_idx] -= diff[still_going] / vega[still_going]

        # Guard against negative or extreme sigma
        bad_sigma = (sigma[update_idx] <= 0) | (sigma[update_idx] > 10)
        bad_sigma_idx = update_idx[bad_sigma]
        sigma[bad_sigma_idx] = np.nan

        # Update active mask
        active[converged_idx] = False
        active[failed_idx] = False
        active[bad_sigma_idx] = False

    return result


# ── Main Processing ────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  IV Calculation for 2024 F&O Data (Vectorized)")
    print("=" * 60)

    # Load data
    print("\n📂 Loading master_fo_data.csv...")
    df = pd.read_csv(INPUT_FILE)
    print(f"   Total rows loaded: {len(df):,}")

    # Filter to 2024 only
    df["TradDt"] = pd.to_datetime(df["TradDt"])
    df_2024 = df[df["TradDt"].dt.year == 2024].copy()
    del df  # free memory
    print(f"   Rows for 2024: {len(df_2024):,}")

    if len(df_2024) == 0:
        print("❌ No 2024 data found!")
        return

    # Filter only option contracts (CE/PE)
    df_2024 = df_2024[df_2024["OptnTp"].isin(["CE", "PE"])].copy()
    print(f"   Option rows (CE/PE): {len(df_2024):,}")

    # Compute time to expiry (T) in years
    df_2024["XpryDt"] = pd.to_datetime(df_2024["XpryDt"])
    df_2024["T"] = (df_2024["XpryDt"] - df_2024["TradDt"]).dt.days / 365.0

    # Filter valid rows
    valid = (
        (df_2024["T"] > 0) &
        (df_2024["ClsPric"] > 0) &
        (df_2024["UndrlygPric"] > 0) &
        (df_2024["StrkPric"] > 0)
    )
    df_2024 = df_2024[valid].copy()
    print(f"   Valid rows (T>0, prices>0): {len(df_2024):,}")

    # Prepare arrays for vectorized computation
    total = len(df_2024)
    print(f"\n🔄 Computing IV for {total:,} rows (vectorized)...")

    market_price = df_2024["ClsPric"].values.astype(np.float64)
    S = df_2024["UndrlygPric"].values.astype(np.float64)
    K = df_2024["StrkPric"].values.astype(np.float64)
    T = df_2024["T"].values.astype(np.float64)
    is_call = (df_2024["OptnTp"].values == "CE")

    # Run vectorized IV
    import time
    start = time.time()
    iv_values = vectorized_iv(market_price, S, K, T, RISK_FREE_RATE, is_call)
    elapsed = time.time() - start

    df_2024["VI"] = iv_values

    # Summary
    valid_count = np.isfinite(iv_values).sum()
    failed_count = total - valid_count
    print(f"   ⏱️  Completed in {elapsed:.1f} seconds")
    print(f"\n📊 IV Calculation Summary:")
    print(f"   Total computed: {total:,}")
    print(f"   Valid IV: {valid_count:,} ({valid_count/total*100:.1f}%)")
    print(f"   Failed/NaN: {failed_count:,} ({failed_count/total*100:.1f}%)")

    # Save
    df_2024.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Saved to: {OUTPUT_FILE}")
    print(f"   Rows: {len(df_2024):,} | Columns: {list(df_2024.columns)}")

    # Sample output
    display_cols = ["TradDt", "TckrSymb", "StrkPric", "OptnTp", "ClsPric", "UndrlygPric", "T", "VI"]
    cols = [c for c in display_cols if c in df_2024.columns]
    print("\n📄 Sample (first 10 rows with valid IV):")
    valid_rows = df_2024[df_2024["VI"].notna()]
    print(valid_rows[cols].head(10).to_string())

    print("\n📄 IV statistics:")
    print(df_2024["VI"].describe().to_string())


if __name__ == "__main__":
    main()
