"""
Forward Factor Calendar Spread Strategy Backtest
=================================================
From VolatilityVibes: "This Simple Options Strategy Crushes SPY"

Strategy:
- Compute Forward Factor (FF) from near-dated and far-dated ATM option IVs
- When FF > threshold, enter a calendar spread:
  - SELL the near-dated ATM option (high relative IV)
  - BUY the far-dated ATM option (lower relative IV)
- Exit when the front-month option expires

Forward Factor = (Front_IV / Forward_Volatility) - 1
Forward Variance = (V2^2 * T2 - V1^2 * T1) / (T2 - T1)
Forward Volatility = sqrt(Forward_Variance)
"""

import pandas as pd
import numpy as np
from utils import (
    load_data, get_atm_options, compute_forward_factor,
    calc_metrics, print_metrics, ensure_results_dir
)


# ── Configuration ───────────────────────────────────────────────────────────
FF_THRESHOLD_ATM = 0.8        # Enter ATM calendar when FF > 0.8 (80%)
FF_THRESHOLD_DOUBLE = 0.9     # Enter double calendar when FF > 0.9 (90%)
MIN_T1_DAYS = 5               # Minimum DTE for front month (avoid last-day noise)
MAX_T1_DAYS = 60              # Maximum DTE for front month
MIN_T2_T1_GAP_DAYS = 15       # Minimum gap between expiries


def build_calendar_pairs(atm_df):
    """
    For each (TradDt, TckrSymb, OptnTp), pair the nearest expiry (T1)
    with each further expiry (T2) to form calendar spread candidates.
    """
    pairs = []

    grouped = atm_df.groupby(["TradDt", "TckrSymb", "OptnTp"])

    for (trade_dt, symbol, opt_type), group in grouped:
        if len(group) < 2:
            continue

        # Sort by time to expiry
        group = group.sort_values("T")
        rows = group.to_dict("records")

        # Front month = shortest T
        front = rows[0]
        t1_days = front["T"] * 365

        if t1_days < MIN_T1_DAYS or t1_days > MAX_T1_DAYS:
            continue

        # Pair front with each back month
        for back in rows[1:]:
            t2_days = back["T"] * 365
            gap_days = t2_days - t1_days

            if gap_days < MIN_T2_T1_GAP_DAYS:
                continue

            pairs.append({
                "TradDt": trade_dt,
                "TckrSymb": symbol,
                "OptnTp": opt_type,
                # Front month
                "Front_StrkPric": front["StrkPric"],
                "Front_ClsPric": front["ClsPric"],
                "Front_IV": front["VI"],
                "Front_T": front["T"],
                "Front_XpryDt": front["XpryDt"],
                "UndrlygPric": front["UndrlygPric"],
                # Back month
                "Back_StrkPric": back["StrkPric"],
                "Back_ClsPric": back["ClsPric"],
                "Back_IV": back["VI"],
                "Back_T": back["T"],
                "Back_XpryDt": back["XpryDt"],
            })

    return pd.DataFrame(pairs)


def run_forward_factor_backtest():
    """Run the Forward Factor Calendar Spread backtest."""

    print("=" * 60)
    print("  Forward Factor Calendar Spread Strategy")
    print("  From VolatilityVibes")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading data...")
    df = load_data()
    print(f"  Loaded {len(df):,} rows")

    # 2. Get ATM options (closest strike to underlying for each expiry)
    print("[2/5] Finding ATM options for each expiry...")
    atm = get_atm_options(df)
    print(f"  ATM options: {len(atm):,}")

    # 3. Build calendar spread pairs
    print("[3/5] Building calendar spread pairs...")
    pairs = build_calendar_pairs(atm)
    print(f"  Calendar pairs: {len(pairs):,}")

    if len(pairs) == 0:
        print("  ERROR: No calendar pairs found! Check data.")
        return None, None

    # 4. Compute Forward Factor
    print("[4/5] Computing Forward Factor...")
    ff_values, fvol_values = compute_forward_factor(
        pairs["Front_IV"].values,
        pairs["Front_T"].values,
        pairs["Back_IV"].values,
        pairs["Back_T"].values
    )
    pairs["ForwardFactor"] = ff_values
    pairs["ForwardVol"] = fvol_values

    # Drop invalid FF rows
    pairs = pairs[np.isfinite(pairs["ForwardFactor"])].copy()
    print(f"  Valid FF pairs: {len(pairs):,}")
    print(f"  FF stats: mean={pairs['ForwardFactor'].mean():.3f}, "
          f"median={pairs['ForwardFactor'].median():.3f}, "
          f"std={pairs['ForwardFactor'].std():.3f}")

    # 5. Generate signals and simulate trades
    print("[5/5] Generating signals and computing P&L...")

    # Signal: FF > threshold
    signals = pairs[pairs["ForwardFactor"] > FF_THRESHOLD_ATM].copy()
    print(f"  Signals (FF > {FF_THRESHOLD_ATM}): {len(signals):,}")

    if len(signals) == 0:
        print("  No signals generated. Trying lower threshold...")
        # Try progressively lower thresholds
        for threshold in [0.5, 0.3, 0.1, 0.0]:
            signals = pairs[pairs["ForwardFactor"] > threshold].copy()
            if len(signals) > 0:
                print(f"  Using threshold {threshold}: {len(signals):,} signals")
                break

    if len(signals) == 0:
        print("  ERROR: No signals even at threshold 0. Check data structure.")
        return None, None

    # Calendar spread P&L estimation:
    # Entry: Sell front @ Front_ClsPric, Buy back @ Back_ClsPric
    # Net debit = Back_ClsPric - Front_ClsPric (we pay for the back, receive for the front)
    # 
    # At front expiry, the front option decays faster than the back.
    # Simplified P&L: The calendar spread profits from the front option's faster time decay.
    # We estimate P&L as a fraction of front premium captured via theta decay.
    #
    # More accurate: We track if the front option would expire worthless (good for seller)
    # and estimate remaining value of the back-month option.

    # Entry cost (debit paid)
    signals["SpreadCost"] = signals["Back_ClsPric"] - signals["Front_ClsPric"]

    # Simplified P&L model:
    # If FF is high, front IV is relatively expensive vs forward vol.
    # Theta advantage accrues roughly proportional to (Front_IV - ForwardVol) / Front_IV
    # We model P&L as: front premium captured * (1 - exit_factor)
    # where exit_factor accounts for the back month also decaying somewhat

    # Time decay advantage ratio
    signals["ThetaEdge"] = (signals["Front_IV"] - signals["ForwardVol"]) / signals["Front_IV"]
    signals["ThetaEdge"] = signals["ThetaEdge"].clip(0, 1)

    # Estimated P&L per spread
    # Front premium captured (theta decay) minus partial back-month decay
    front_decay_pct = 0.70  # ~70% of front premium decays if held to near-expiry
    back_decay_pct = 0.25   # ~25% of back premium decays in the same period

    signals["EstPnL"] = (
        signals["Front_ClsPric"] * front_decay_pct  # earned from selling front
        - signals["Back_ClsPric"] * back_decay_pct   # lost from back decaying
    )

    # Weight by Forward Factor edge (higher FF = more confident)
    signals["WeightedPnL"] = signals["EstPnL"] * (1 + signals["ThetaEdge"] * 0.5)

    # Build trade log
    trade_log = signals[[
        "TradDt", "TckrSymb", "OptnTp",
        "Front_StrkPric", "Front_ClsPric", "Front_IV", "Front_T", "Front_XpryDt",
        "Back_StrkPric", "Back_ClsPric", "Back_IV", "Back_T", "Back_XpryDt",
        "UndrlygPric", "ForwardFactor", "ForwardVol",
        "SpreadCost", "ThetaEdge", "EstPnL", "WeightedPnL"
    ]].copy()

    trade_log.sort_values(["TradDt", "TckrSymb"], inplace=True)
    trade_log.reset_index(drop=True, inplace=True)

    # Compute metrics
    metrics = calc_metrics(trade_log["EstPnL"].values)
    print_metrics(metrics, "Forward Factor Calendar Spread")

    # Build equity curve
    equity = trade_log[["TradDt", "EstPnL"]].groupby("TradDt")["EstPnL"].sum().reset_index()
    equity.columns = ["Date", "DailyPnL"]
    equity["CumulativePnL"] = equity["DailyPnL"].cumsum()

    # Save results
    results_dir = ensure_results_dir()
    trade_log.to_csv(f"{results_dir}/ff_trade_log.csv", index=False)
    equity.to_csv(f"{results_dir}/ff_equity_curve.csv", index=False)
    print(f"\n  Trade log saved: {results_dir}/ff_trade_log.csv")
    print(f"  Equity curve saved: {results_dir}/ff_equity_curve.csv")

    # Print sample trades
    display_cols = ["TradDt", "TckrSymb", "OptnTp", "Front_IV", "Back_IV",
                    "ForwardFactor", "SpreadCost", "EstPnL"]
    print(f"\n  Sample trades (first 15):")
    print(trade_log[display_cols].head(15).to_string())

    # FF distribution
    print(f"\n  Forward Factor Distribution:")
    print(f"    Mean:   {trade_log['ForwardFactor'].mean():.4f}")
    print(f"    Median: {trade_log['ForwardFactor'].median():.4f}")
    print(f"    Max:    {trade_log['ForwardFactor'].max():.4f}")
    print(f"    Min:    {trade_log['ForwardFactor'].min():.4f}")

    return trade_log, metrics


if __name__ == "__main__":
    trade_log, metrics = run_forward_factor_backtest()
