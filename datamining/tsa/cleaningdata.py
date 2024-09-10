import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
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
fig, ax = plt.subplots(2, 1, figsize=(12,8))

plot_acf(df['Temperature'], ax=ax[0], lags=100)
ax[0].set_title('ACF - Temperature')

plot_pacf(df['Temperature'], ax=ax[1], lags=100)
ax[1].set_title('PACF - Temperature')

plt.tight_layout()
plt.savefig('PACF-ACF_Temperature.png')

# Plot ACF and PACF for Humidity
fig, ax = plt.subplots(2, 1, figsize=(12,8))

plot_acf(df['Humidity'], ax=ax[0], lags=100)
ax[0].set_title('ACF - Humidity')

plot_pacf(df['Humidity'], ax=ax[1], lags=100)
ax[1].set_title('PACF - Humidity')

plt.tight_layout()
plt.savefig('PACF-ACF_Humidity.png')



start_date = '2024-04-14 19:00:00'
end_date = '2024-04-19 21:00:00'
intervals = pd.date_range(start=start_date, end=end_date, freq='24h')

df['Temperature_diff'] = df['Temperature'].diff().dropna()

train_df = df[df.index < intervals[4]]  # First 5 days
test_df = df[(df.index >= intervals[4]) & (df.index < intervals[5])]  # 6th day

# Initialize lists to store forecasted values
forecasted_temperature = []
forecasted_humidity = []

# Train ARIMA model for Temperature
model_temp = ARIMA(train_df['Temperature'], order=(2, 1, 0))  # Adjust params as needed
model_temp_fit = model_temp.fit()

# Train SARIMA model for Humidity
model_humidity = SARIMAX(train_df['Humidity'], order=(2, 0, 0), seasonal_order=(2, 1, 0, 96))  # Adjusted seasonal order
model_humidity_fit = model_humidity.fit()

# Forecasting hour by hour and updating the model
for hour in pd.date_range(start=intervals[4], end=intervals[4] + pd.Timedelta(hours=23), freq='h'):
    # Forecast one hour ahead for both temperature and humidity
    forecast_temp = model_temp_fit.forecast(steps=1)[0]
    forecast_humidity = model_humidity_fit.forecast(steps=1)[0]
    
    # Store forecasts
    forecasted_temperature.append(forecast_temp)
    forecasted_humidity.append(forecast_humidity)
    
    # Update models with actual data from the test set (if available)
    if hour in test_df.index:
        actual_temp = test_df['Temperature'].loc[hour]
        actual_humidity = test_df['Humidity'].loc[hour]
        
        # Update the ARIMA model for temperature with new data
        model_temp = ARIMA(train_df['Temperature'].append(pd.Series(actual_temp, index=[hour])), order=(2, 1, 0))
        model_temp_fit = model_temp.fit()
        
        # Update the SARIMA model for humidity with new data
        model_humidity = SARIMAX(train_df['Humidity'].append(pd.Series(actual_humidity, index=[hour])), 
                                 order=(2, 0, 0), seasonal_order=(2, 1, 0, 96))
        model_humidity_fit = model_humidity.fit()

# Create DataFrame for forecasted values
forecast_df = pd.DataFrame({
    'Forecasted Temperature': forecasted_temperature,
    'Forecasted Humidity': forecasted_humidity
}, index=pd.date_range(start=intervals[4], end=intervals[4] + pd.Timedelta(hours=23), freq='h'))

# Plotting the forecasts and actual data
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(forecast_df['Forecasted Temperature'], label='Forecasted Temperature', color='blue')
plt.plot(test_df['Temperature'], label='Actual Temperature', color='orange')
plt.title("Temperature Forecast vs Actual for 6th Day")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(forecast_df['Forecasted Humidity'], label='Forecasted Humidity', color='green')
plt.plot(test_df['Humidity'], label='Actual Humidity', color='red')
plt.title("Humidity Forecast vs Actual for 6th Day")
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig('Forecast.png')
# df['interval'] = pd.cut(df.index, bins=intervals, labels=intervals[:-1])
# #print(df_daily)
# #sns.lineplot(df_daily)
# df_daily = [group for _, group in df.groupby('interval', observed=False)]
# colors = plt.cm.jet(np.linspace(0, 1, len(df_daily)))  # Generate colors for the days

# plt.figure(figsize=(12, 6))
# for i, day_data in enumerate(df_daily):
#     hours = day_data.index.hour
#     minutes = day_data.index.minute
#     time = hours + minutes / 60.0
#     time = np.where(time > 19, time - 20, time + 24 - 20) 
#     plt.plot(time, day_data['Temperature'], label=f'Date: {day_data.index.date[0]}', color=colors[i])


# plt.title('Temperature over 24 Hours')
# plt.xlabel('Hour of Day (starting from 19)')
# plt.ylabel('Temperature (°C)')
# plt.xticks(np.arange(0, 24, 1))
# plt.xlim(0, 24)  # Set x-limits to cover the full range from 0 to 23
# plt.grid(True)  # Optional, to make the plot easier to read
# plt.legend(loc='upper right', fontsize='small')
# plt.savefig('output/overlap_temps_after.png')

# plt.figure(figsize=(12, 6))
# for i, day_data in enumerate(df_daily):
#     hours = day_data.index.hour
#     minutes = day_data.index.minute
#     time = hours + minutes / 60.0
#     time = np.where(time > 19, time - 20, time + 24 - 20) 
#     plt.plot(time, day_data['Humidity'], label=f'Date: {day_data.index.date[0]}', color=colors[i])

# plt.title('Humidity over 24 Hours')
# plt.xlabel('Hour of day')
# plt.ylabel('Humidity (%)')
# plt.xticks(np.arange(0, 24, 1))
# plt.xlim(0, 24)  # Set x-limits to cover the full range from 19 to 18 (next day)
# plt.grid(True)  # Optional, to make the plot easier to read
# plt.legend(loc='upper right', fontsize='small')
# plt.savefig('output/overlap_humidity_after.png')


# plt.figure(figsize=(12, 6))
# plt.plot(df.index, df['Temperature'], label='Complete Temperature Data', color='blue')

# start_of_day = df.index[df.index.hour == pd.timedelta_range]
# for day in intervals:
#     plt.axvline(day, color='k', linestyle='--', alpha=0.2)
# plt.axhline(df['Temperature'].mean(), color='r', alpha=0.3, linestyle='--')
# plt.title('Temperature Over 24 Hours (Complete Data)')
# plt.xlabel('Time')
# plt.ylabel('Temperature (°C)')
# plt.legend(loc='upper right', fontsize='small')
# plt.grid(True)
# plt.savefig('output/all_time_temps_after.png')


# plt.figure(figsize=(12, 6))
# plt.plot(df.index, df['Humidity'], label='Complete Humidity Data', color='red')

# start_of_day = df.index[df.index.hour == 0]
# for day in intervals:
#     plt.axvline(day, color='k', linestyle='--', alpha=0.2)
# plt.axhline(df['Humidity'].mean(), color='b', alpha=0.3, linestyle='--')
# plt.title('Humidity Over 24 Hours (Complete Data)')
# plt.xlabel('Time')
# plt.ylabel('Humidity (%)')
# plt.legend(loc='upper right', fontsize='small')
# plt.grid(True)
# plt.savefig('output/all_time_humidity_after.png')
