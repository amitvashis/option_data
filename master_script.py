import pandas as pd
import os
import glob

# Paths
clean_dir = "D:/antigravity/option_data/clean_data/"
master_dir = "D:/antigravity/option_data/master/"
master_file = os.path.join(master_dir, "master_fo_data.csv")

os.makedirs(master_dir, exist_ok=True)

# Find all processed CSV files
csv_files = sorted(glob.glob(os.path.join(clean_dir, "processed_fo*.csv")))
print(f"Found {len(csv_files)} processed CSV files.\n")

# Read and append all into a single DataFrame
all_dfs = []
for i, f in enumerate(csv_files, 1):
    basename = os.path.basename(f)
    df = pd.read_csv(f)
    all_dfs.append(df)
    print(f"[{i}/{len(csv_files)}] Loaded {basename} — {len(df)} rows")

# Concatenate and save
master_df = pd.concat(all_dfs, ignore_index=True)
master_df.to_csv(master_file, index=False)
print(f"\nMaster file saved: {master_file}")
print(f"Total rows: {len(master_df)}, Columns: {len(master_df.columns)}")
