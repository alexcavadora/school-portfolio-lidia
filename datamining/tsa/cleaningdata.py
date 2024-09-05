import pandas as pd
file = 'dataset/DATASET.CSV'
#df_split = [pd.read_csv(file + f"{path}.CSV") for path in range(6)]
#df = pd.concat(df_split, ignore_index=True)
df = (pd.read_csv(file))
df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y %H:%M:%S')
df.set_index('DATE', inplace=True)
df.sort_index(inplace=True)

print(df)
df_resampled = df.resample('h').mean()  # Resample by hour and calculate the mean
print(df_resampled)
