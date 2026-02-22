"""
Backtest Runner — Runs all option strategies and prints combined results.
=========================================================================
From VolatilityVibes YouTube Channel Strategies

Usage:
    python run_backtest.py
"""

import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

# Add backtesting dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from forward_factor_strategy import run_forward_factor_backtest
from iv_rank_strategy import run_iv_rank_backtest
from utils import ensure_results_dir


def main():
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#    VolatilityVibes Option Strategy Backtester" + " " * 22 + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    results_dir = ensure_results_dir()

    # ── Strategy 1: Forward Factor Calendar Spread ──────────────────────
    print("\n\n" + "=" * 70)
    print("  STRATEGY 1: FORWARD FACTOR CALENDAR SPREAD")
    print("=" * 70)

    try:
        ff_trades, ff_metrics = run_forward_factor_backtest()
    except Exception as e:
        print(f"  ERROR running Forward Factor strategy: {e}")
        import traceback
        traceback.print_exc()
        ff_trades, ff_metrics = None, None

    # ── Strategy 2: IV Rank Long Straddle ───────────────────────────────
    print("\n\n" + "=" * 70)
    print("  STRATEGY 2: IV RANK LONG STRADDLE")
    print("=" * 70)

    try:
        ivr_trades, ivr_metrics = run_iv_rank_backtest()
    except Exception as e:
        print(f"  ERROR running IV Rank strategy: {e}")
        import traceback
        traceback.print_exc()
        ivr_trades, ivr_metrics = None, None

    # ── Combined Summary ────────────────────────────────────────────────
    print("\n\n" + "#" * 70)
    print("  COMBINED BACKTEST SUMMARY")
    print("#" * 70)

    strategies = [
        ("Forward Factor Calendar", ff_metrics),
        ("IV Rank Straddle", ivr_metrics),
    ]

    # Header
    print(f"\n  {'Strategy':<30} {'Trades':>8} {'Total PnL':>12} {'Win Rate':>10} "
          f"{'Sharpe':>8} {'Max DD':>10} {'Profit F':>10}")
    print("  " + "-" * 90)

    for name, m in strategies:
        if m is None:
            print(f"  {name:<30} {'ERROR':>8}")
            continue
        print(f"  {name:<30} {m['total_trades']:>8} {m['total_pnl']:>12.2f} "
              f"{m['win_rate']:>9.1f}% {m['sharpe']:>8.2f} "
              f"{m['max_drawdown_pct']:>9.2f}% {m['profit_factor']:>10.2f}")

    print(f"\n  Results saved to: {os.path.abspath(results_dir)}")
    print(f"\n  Files generated:")
    for f in os.listdir(results_dir):
        fpath = os.path.join(results_dir, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    - {f} ({size_kb:.1f} KB)")

    print("\n" + "#" * 70)
    print("  Backtest complete!")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
