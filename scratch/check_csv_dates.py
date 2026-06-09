import pandas as pd
df = pd.read_csv("data/historical_xauusdm_30.csv")
print("First 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())
print("\nDate range:")
print(f"Min time: {df['time'].min()}")
print(f"Max time: {df['time'].max()}")
