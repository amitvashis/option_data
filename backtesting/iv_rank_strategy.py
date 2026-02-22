"""
IV Rank / Percentile Strategy Backtest
=======================================
Inspired by VolatilityVibes: "The Earnings Volatility Strategy Big Funds Use"

Strategy:
- Track IV percentile rank for each symbol over a rolling window
- Enter LONG STRADDLE when IV rank is low (< 20) — IV likely to expand
- Exit when IV rank rises above 50 — IV expansion captured
- Alternative exit: max holding period of 15 trading days

IV Rank = (Current_IV - Min_IV) / (Max_IV - Min_IV) * 100
"""

import pandas as pd
import numpy as np
from utils import (
    load_data, get_atm_options,
    calc_metrics, print_metrics, ensure_results_dir
)


# ── Configuration ───────────────────────────────────────────────────────────
IV_RANK_ENTRY = 20        # Enter when IV rank < 20 (low IV environment)
IV_RANK_EXIT = 50         # Exit when IV rank > 50 (IV has expanded)
MAX_HOLDING_DAYS = 15     # Max holding period (trading days)
ROLLING_WINDOW = 20       # Rolling window for IV rank (trading days)
MIN_WINDOW_DATA = 10      # Minimum data points for rank calculation


def compute_iv_rank(iv_series, window=ROLLING_WINDOW, min_periods=MIN_WINDOW_DATA):
    """
    Compute rolling IV percentile rank.
    IV Rank = (Current - Rolling_Min) / (Rolling_Max - Rolling_Min) * 100
    """
    iv_series = pd.Series(iv_series).reset_index(drop=True)
    rolling_min = iv_series.rolling(window=window, min_periods=min_periods).min()
    rolling_max = iv_series.rolling(window=window, min_periods=min_periods).max()

    iv_range = rolling_max - rolling_min
    iv_rank = np.where(iv_range > 1e-8,
                       (iv_series - rolling_min) / iv_range * 100,
                       50)  # default to 50 if no range
    return iv_rank


def run_iv_rank_backtest():
    """Run the IV Rank Long Straddle backtest."""

    print("=" * 60)
    print("  IV Rank Long Straddle Strategy")
    print("  Inspired by VolatilityVibes")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading data...")
    df = load_data()
    print(f"  Loaded {len(df):,} rows")

    # 2. Get ATM options — nearest expiry only
    print("[2/5] Finding ATM options (nearest expiry)...")
    atm = get_atm_options(df)

    # Keep only nearest expiry per (TradDt, TckrSymb, OptnTp)
    atm_nearest = atm.sort_values("T")
    atm_nearest = atm_nearest.groupby(["TradDt", "TckrSymb", "OptnTp"]).first().reset_index()
    print(f"  ATM nearest-expiry rows: {len(atm_nearest):,}")

    # 3. Build straddles: merge CE and PE for each (TradDt, TckrSymb)
    print("[3/5] Building ATM straddles...")
    ce = atm_nearest[atm_nearest["OptnTp"] == "CE"].copy()
    pe = atm_nearest[atm_nearest["OptnTp"] == "PE"].copy()

    straddles = ce.merge(pe, on=["TradDt", "TckrSymb"], suffixes=("_CE", "_PE"))

    if len(straddles) == 0:
        print("  ERROR: No straddles formed. Check data.")
        return None, None

    straddles["StraddleCost"] = straddles["ClsPric_CE"] + straddles["ClsPric_PE"]
    straddles["AvgIV"] = (straddles["VI_CE"] + straddles["VI_PE"]) / 2
    straddles["UndrlygPric"] = straddles["UndrlygPric_CE"]

    print(f"  Straddles formed: {len(straddles):,}")

    # 4. Compute IV Rank for each symbol
    print("[4/5] Computing IV Rank per symbol...")
    straddles.sort_values(["TckrSymb", "TradDt"], inplace=True)

    iv_rank_all = []
    for symbol, sdf in straddles.groupby("TckrSymb"):
        sdf = sdf.sort_values("TradDt").copy()
        sdf["IVRank"] = compute_iv_rank(sdf["AvgIV"].values)
        iv_rank_all.append(sdf)

    straddles = pd.concat(iv_rank_all, ignore_index=True)
    straddles = straddles[straddles["IVRank"].notna()].copy()
    print(f"  Rows with valid IV rank: {len(straddles):,}")

    # 5. Generate signals and simulate trades
    print("[5/5] Generating signals and computing P&L...")

    trades = []
    # Track open positions per symbol
    open_positions = {}  # symbol -> entry row

    straddles.sort_values("TradDt", inplace=True)

    for _, row in straddles.iterrows():
        symbol = row["TckrSymb"]
        trade_dt = row["TradDt"]
        iv_rank = row["IVRank"]

        # Check for exit
        if symbol in open_positions:
            entry = open_positions[symbol]
            days_held = (trade_dt - entry["TradDt"]).days

            should_exit = (iv_rank > IV_RANK_EXIT) or (days_held >= MAX_HOLDING_DAYS)

            if should_exit:
                # P&L: change in straddle value
                # If IV expanded, straddle value goes up
                exit_straddle = row["StraddleCost"]
                entry_straddle = entry["StraddleCost"]

                # Also account for underlying price movement
                # Straddle profits if price moves enough
                underly_move = abs(row["UndrlygPric"] - entry["UndrlygPric"])
                move_pct = underly_move / entry["UndrlygPric"] if entry["UndrlygPric"] > 0 else 0

                # P&L = exit value - entry cost
                # Straddle value at exit includes both IV change and price movement
                pnl = exit_straddle - entry_straddle

                # If IV expanded significantly, add the vega gain
                iv_change = row["AvgIV"] - entry["AvgIV"]

                trades.append({
                    "EntryDate": entry["TradDt"],
                    "ExitDate": trade_dt,
                    "TckrSymb": symbol,
                    "EntryIVRank": entry["IVRank"],
                    "ExitIVRank": iv_rank,
                    "EntryIV": entry["AvgIV"],
                    "ExitIV": row["AvgIV"],
                    "IVChange": iv_change,
                    "EntryStraddleCost": entry_straddle,
                    "ExitStraddleValue": exit_straddle,
                    "UndrlygMove_Pct": round(move_pct * 100, 2),
                    "DaysHeld": days_held,
                    "PnL": round(pnl, 2),
                    "PnL_Pct": round(pnl / entry_straddle * 100, 2) if entry_straddle > 0 else 0,
                    "ExitReason": "IV_Expansion" if iv_rank > IV_RANK_EXIT else "MaxDays"
                })

                del open_positions[symbol]

        # Check for entry (only if not already in a position)
        if symbol not in open_positions and iv_rank < IV_RANK_ENTRY:
            if row["StraddleCost"] > 0:
                open_positions[symbol] = row.to_dict()

    if len(trades) == 0:
        print("  No trades generated. Adjusting thresholds...")
        # Show IV Rank distribution
        print(f"  IV Rank distribution:")
        print(f"    Min: {straddles['IVRank'].min():.1f}")
        print(f"    Max: {straddles['IVRank'].max():.1f}")
        print(f"    Mean: {straddles['IVRank'].mean():.1f}")
        print(f"    < 20: {(straddles['IVRank'] < 20).sum()}")
        print(f"    < 30: {(straddles['IVRank'] < 30).sum()}")
        print(f"    < 40: {(straddles['IVRank'] < 40).sum()}")
        return None, None

    trade_log = pd.DataFrame(trades)
    trade_log.sort_values("EntryDate", inplace=True)
    trade_log.reset_index(drop=True, inplace=True)

    # Compute metrics
    metrics = calc_metrics(trade_log["PnL"].values)
    print_metrics(metrics, "IV Rank Long Straddle")

    # Build equity curve
    equity = trade_log[["ExitDate", "PnL"]].groupby("ExitDate")["PnL"].sum().reset_index()
    equity.columns = ["Date", "DailyPnL"]
    equity["CumulativePnL"] = equity["DailyPnL"].cumsum()

    # Save results
    results_dir = ensure_results_dir()
    trade_log.to_csv(f"{results_dir}/ivr_trade_log.csv", index=False)
    equity.to_csv(f"{results_dir}/ivr_equity_curve.csv", index=False)
    print(f"\n  Trade log saved: {results_dir}/ivr_trade_log.csv")
    print(f"  Equity curve saved: {results_dir}/ivr_equity_curve.csv")

    # Print sample trades
    display_cols = ["EntryDate", "TckrSymb", "EntryIVRank", "ExitIVRank",
                    "EntryIV", "ExitIV", "DaysHeld", "PnL", "ExitReason"]
    print(f"\n  Sample trades (first 15):")
    print(trade_log[display_cols].head(15).to_string())

    # Summary by exit reason
    print(f"\n  Exit Reason Breakdown:")
    for reason, grp in trade_log.groupby("ExitReason"):
        print(f"    {reason}: {len(grp)} trades, avg PnL={grp['PnL'].mean():.2f}")

    return trade_log, metrics


if __name__ == "__main__":
    trade_log, metrics = run_iv_rank_backtest()
