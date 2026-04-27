import pandas as pd
import os

DATA_PATH = r"C:\Users\suraj\Intern Project\Dataset\cert-dataset\r4.2"

log_files = ['logon.csv', 'device.csv', 'email.csv', 'file.csv', 'http.csv']

for fname in log_files:
    path = os.path.join(DATA_PATH, fname)
    df = pd.read_csv(path, nrows=5000)
    print(f"\n{'='*50}")
    print(f"FILE: {fname}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Dtypes:\n{df.dtypes}")
    print(f"Sample:\n{df.head(3)}")
    print(f"Nulls:\n{df.isnull().sum()}")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"Unique users: {df['user'].nunique()}")
