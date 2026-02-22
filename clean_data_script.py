import pandas as pd
import os
import glob

# Paths
raw_dir = "D:/Antigravity/option_data/"
clean_dir = "D:/Antigravity/option_data/clean_data/"

os.makedirs(clean_dir, exist_ok=True)


def clean_bhavcopy(input_path, output_path):
    """Clean a single F&O bhavcopy CSV file.

    Steps:
    1. Drop columns that are completely empty (all NaN)
    2. Drop rows that have any null value in remaining columns
    3. Drop rows that have 0 in any numeric column
    """
    df = pd.read_csv(input_path)
    original_shape = df.shape

    # Step 1: Drop columns that are completely empty
    empty_cols = [col for col in df.columns if df[col].isnull().all()]
    df = df.drop(columns=empty_cols)
    print(f"  Dropped {len(empty_cols)} empty columns: {empty_cols}")

    # Step 2: Drop rows with any null values
    null_rows = df.isnull().any(axis=1).sum()
    df = df.dropna()
    print(f"  Dropped {null_rows} rows with null values")

    # Step 3: Drop rows with 0 in any numeric column
    numeric_cols = df.select_dtypes(include="number").columns
    zero_mask = (df[numeric_cols] == 0).any(axis=1)
    zero_rows = zero_mask.sum()
    df = df[~zero_mask]
    print(f"  Dropped {zero_rows} rows with 0 values in numeric columns")

    # Save
    df.to_csv(output_path, index=False)
    print(f"  Result: {original_shape} -> {df.shape}")
    print(f"  Saved to: {output_path}")
    return df


# Process all CSV files in option_data folder
csv_files = sorted(glob.glob(os.path.join(raw_dir, "fo*.csv")))
print(f"Found {len(csv_files)} CSV files to process.\n")

for i, input_file in enumerate(csv_files, 1):
    basename = os.path.basename(input_file)
    output_file = os.path.join(clean_dir, f"processed_{basename}")

    # Skip if already processed
    if os.path.isfile(output_file):
        print(f"[{i}/{len(csv_files)}] Skipping (already exists): {basename}")
        continue

    print(f"[{i}/{len(csv_files)}] Processing: {basename}")
    try:
        clean_bhavcopy(input_file, output_file)
    except Exception as e:
        print(f"  ERROR: {e}")
    print()

print("All done!")
