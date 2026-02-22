"""
Filter VI data: For each (TradDt, TckrSymb, OptnTp),
pick the ONE row where StrkPric is closest to UndrlygPric.
Saves result as atm_vi_data.csv
"""

import pandas as pd
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ── Configuration ───────────────────────────────────────────────────────────
INPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vi_data_historical.csv")
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "atm_vi_data.csv")

# ── Main ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Filter ATM Options — Closest Strike to Underlying")
print("  One entry per (Date, Symbol, CE/PE)")
print("=" * 60)

print(f"\n📂 Loading data...")
df = pd.read_csv(INPUT_FILE)
print(f"   Total rows: {len(df):,}")

# Only keep CE and PE rows
df = df[df["OptnTp"].isin(["CE", "PE"])].copy()
print(f"   Option rows (CE/PE): {len(df):,}")

# Calculate absolute distance from strike to underlying
df["_dist"] = abs(df["StrkPric"] - df["UndrlygPric"])

# For each (TradDt, TckrSymb, OptnTp), pick the row with minimum distance
idx = df.groupby(["TradDt", "TckrSymb", "OptnTp"])["_dist"].idxmin()
atm = df.loc[idx].copy()

# Drop helper column
atm.drop(columns=["_dist"], inplace=True)

# Sort
atm.sort_values(["TradDt", "TckrSymb", "OptnTp", "StrkPric"], inplace=True)
atm.reset_index(drop=True, inplace=True)

# Save
atm.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ ATM rows selected: {len(atm):,}")
print(f"   Unique symbols: {atm['TckrSymb'].nunique()}")
print(f"   Unique dates: {atm['TradDt'].nunique()}")
print(f"\n💾 Saved to: {OUTPUT_FILE}")

# Display sample
cols = ["TradDt", "TckrSymb", "StrkPric", "OptnTp", "ClsPric", "UndrlygPric", "VI"]
cols = [c for c in cols if c in atm.columns]
print(f"\n📄 Sample (first 20 rows):")
print(atm[cols].head(20).to_string())

print(f"\n📊 VI Statistics:")
print(atm["VI"].describe().to_string())
