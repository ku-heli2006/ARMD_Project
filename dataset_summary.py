import pandas as pd
import os

df = pd.read_parquet('data/armd_base_joined.parquet')

rows, cols = df.shape
memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
file_size_mb = os.path.getsize('data/armd_base_joined.parquet') / (1024 ** 2)

print("=" * 40)
print("DATASET SUMMARY")
print("=" * 40)
print(f"Rows (culture orders):     {rows:,}")
print(f"Columns (features):       {cols:,}")
print(f"Total cells:               {rows * cols:,}")
print(f"In-memory size (pandas):  {memory_mb:,.2f} MB")
print(f"File size on disk (Parquet): {file_size_mb:,.2f} MB")
print("=" * 40)