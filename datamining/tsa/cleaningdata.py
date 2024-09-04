import pandas as pd
file = 'dataset/DATASET'
df_split = [pd.read_csv(file + f"{path}.CSV") for path in range(6)]
df = pd.concat(df_split, ignore_index=True)
df['DATE'] = pd.to_datetime(df['DATE'], format='mixed', dayfirst=True)
df.set_index('DATE', inplace=True)
df.sort_index(inplace=True)
print(df)