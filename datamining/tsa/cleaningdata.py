import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
def adf_test(series, title=''):
    """
    Pass in a time series and an optional title, returns ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(), autolag='AIC')
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    out = pd.Series(result[0:4], index=labels)
    for key, value in result[4].items():
        out[f'Critical Value ({key})'] = value
    print(out.to_string())
    if result[1] <= 0.05:
        print("=> The series is stationary.")
    else:
        print("=> The series is not stationary.")
    print()

file = 'dataset/DATASET.csv'
df = (pd.read_csv(file))

#checando la fecha para convertirla a el nuevo index, l;uego se ordenan para evitar cualquier cosa
df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y %H:%M:%S')
df.set_index('DATE', inplace=True)

#promediamos el tiempo para igualar los intervalos de datos
df = df.resample('15min').mean() 
df.sort_index(inplace=True)
#adf_test(df['Temperature'], 'Temperature')
#adf_test(df['Humidity'], 'Humidity')
#df = df.diff().dropna()
#adf_test(df['Temperature'], 'Temperature')
#adf_test(df['Humidity'], 'Humidity')
#print(df)
start_date = '2024-04-14 19:00:00'
end_date = '2024-04-19 21:00:00'
intervals = pd.date_range(start=start_date, end=end_date, freq='24h')
df['interval'] = pd.cut(df.index, bins=intervals, labels=intervals[:-1])
#print(df_daily)
#sns.lineplot(df_daily)
df_daily = [group for _, group in df.groupby('interval', observed=False)]
colors = plt.cm.jet(np.linspace(0, 1, len(df_daily)))  # Generate colors for the days

plt.figure(figsize=(12, 6))
for i, day_data in enumerate(df_daily):
    hours = day_data.index.hour
    minutes = day_data.index.minute
    time = hours + minutes / 60.0
    time = np.where(time > 19, time - 20, time + 24 - 20) 
    plt.plot(time, day_data['Temperature'], label=f'Date: {day_data.index.date[0]}', color=colors[i])


plt.title('Temperature over 24 Hours')
plt.xlabel('Hour of Day (starting from 19)')
plt.ylabel('Temperature (°C)')
plt.xticks(np.arange(0, 24, 1))
plt.xlim(0, 24)  # Set x-limits to cover the full range from 0 to 23
plt.grid(True)  # Optional, to make the plot easier to read
plt.legend(loc='upper right', fontsize='small')
plt.savefig('output/overlap_temps_before.png')

plt.figure(figsize=(12, 6))
for i, day_data in enumerate(df_daily):
    hours = day_data.index.hour
    minutes = day_data.index.minute
    time = hours + minutes / 60.0
    time = np.where(time > 19, time - 20, time + 24 - 20) 
    plt.plot(time, day_data['Humidity'], label=f'Date: {day_data.index.date[0]}', color=colors[i])

plt.title('Humidity over 24 Hours')
plt.xlabel('Hour of day')
plt.ylabel('Humidity (%)')
plt.xticks(np.arange(0, 24, 1))
plt.xlim(0, 24)  # Set x-limits to cover the full range from 19 to 18 (next day)
plt.grid(True)  # Optional, to make the plot easier to read
plt.legend(loc='upper right', fontsize='small')
plt.savefig('output/overlap_humidity_before.png')



# Now for the complete data (handling missing values)
# We will use forward-fill or you can interpolate depending on your preference
#df_filled = df.ffill()  # Forward-fill missing data (you can use interpolation if needed)

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Temperature'], label='Complete Temperature Data', color='blue')

# Add vertical lines at the start of each day
start_of_day = df.index[df.index.hour == 0]
for timestamp in start_of_day:
    plt.axvline(x=timestamp, color='red', linestyle='-', linewidth=0.5)

plt.title('Temperature Over 24 Hours (Complete Data)')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.legend(loc='upper right', fontsize='small')
plt.grid(True)
plt.savefig('output/all_time_temps_before.png')


# Add vertical lines at the start of each day
plt.figure(figsize=(24, 12))
plt.plot(df.index, df['Humidity'], label='Complete Humidity Data', color='green')

# Add vertical lines at the start of each day
start_of_day = df.index[df.index.hour == 0]
for timestamp in start_of_day:
    plt.axvline(x=timestamp, color='red', linestyle='-', linewidth=0.5)

plt.title('Humidity Over 24 Hours (Complete Data)')
plt.xlabel('Time')
plt.ylabel('Humidity (%)')
plt.legend(loc='upper right', fontsize='small')
plt.grid(True)
plt.savefig('output/all_time_humidity_before.png')
