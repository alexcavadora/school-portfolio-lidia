import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

p_values = [i for i in range(10)]
d_values = [i for i in range(2)]
q_values = [i for i in range(10)]

def evaluate_arima_model(train_data, test_data, arima_order):
    model = ARIMA(train_data, order=arima_order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test_data))
    mse = root_mean_squared_error(test_data, forecast)
    return mse

def grid_search_arima(train_df, test_df, target_column):
    best_score, best_cfg = 1000, None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    arima_order = (p, d, q)
                    mse = evaluate_arima_model(train_df[target_column], test_df[target_column], arima_order)
                    if mse < best_score:
                        best_score, best_cfg = mse, arima_order
                    print(f'{target_column}: ARIMA{arima_order} MSE={mse}')
                except Exception as e:
                    print(f'ARIMA{arima_order} failed')
                    continue
    return best_cfg

def adf_test(series, title=''):
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(), autolag='AIC')
    labels = ['ADF Test Statistic', 'p-value','# of lags']
    out = pd.Series(result[0:4], index=labels)
    for key, value in result[4].items():
        out[f'Critical Value ({key})'] = value
    print(out.to_string())
    if result[1] <= 0.05:
        print("=> The series is stationary.")
    else:
        print("=> The series is not stationary.")
    print()

def mean_absolute_error(actual, predicted):
    return np.mean(np.abs(actual - predicted))

def root_mean_squared_error(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

file = 'dataset/DATASET.csv'
df = (pd.read_csv(file))
df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y %H:%M:%S')
df.set_index('DATE', inplace=True)
df = df.resample('15min').mean()
df.sort_index(inplace=True)
start_date = '2024-04-14 19:00:00'
end_date = '2024-04-19 21:00:00'
intervals = pd.date_range(start=start_date, end=end_date, freq='24h')
train_df = df[df.index < intervals[4]]
test_df = df[(df.index >= intervals[4]) & (df.index < intervals[5])]
forecasted_temperature = []
forecasted_humidity = []
temp_params = [1,0,0]
humidity_params = [1,0,0]

model_temp = ARIMA(train_df['Temperature'], order=temp_params)
model_temp_fit = model_temp.fit()

model_humidity = ARIMA(train_df['Humidity'], order=humidity_params)
model_humidity_fit = model_humidity.fit()

for time in pd.date_range(start=intervals[4] - pd.Timedelta(minutes=15) , end=intervals[5] -  pd.Timedelta(minutes=30), freq='15min'):
    forecast_temp = model_temp_fit.forecast(steps=1).iloc[0]
    forecast_humidity = model_humidity_fit.forecast(steps=1).iloc[0]

    forecasted_temperature.append(forecast_temp)
    forecasted_humidity.append(forecast_humidity)

    if time in test_df.index:
        actual_temp = test_df['Temperature'].loc[time]
        actual_humidity = test_df['Humidity'].loc[time]


        temp_series = pd.concat([train_df['Temperature'], pd.Series(actual_temp, index=[time])])
        model_temp = ARIMA(temp_series, order=temp_params)
        model_temp_fit = model_temp.fit()


        humidity_series = pd.concat([train_df['Humidity'], pd.Series(actual_humidity, index=[time])])
        model_humidity = ARIMA(humidity_series, order=humidity_params)
        model_humidity_fit = model_humidity.fit()

naive_forecast_temp = []
naive_forecast_humidity = []

last_temp = train_df['Temperature'].iloc[-1]
last_humidity = train_df['Humidity'].iloc[-1]



for idx in test_df.index:
    naive_forecast_temp.append(last_temp)
    naive_forecast_humidity.append(last_humidity)
    last_temp = test_df['Temperature'].loc[idx]
    last_humidity = test_df['Humidity'].loc[idx]
forecasted_temp_array = np.array(forecasted_temperature)
forecasted_humidity_array = np.array(forecasted_humidity)

actual_temp_array = test_df['Temperature']
actual_humidity_array = test_df['Humidity']

naive_temp_array = np.array(naive_forecast_temp)
naive_humidity_array = np.array(naive_forecast_humidity)
mae_temp_model = mean_absolute_error(actual_temp_array, forecasted_temp_array)
rmse_temp_model = root_mean_squared_error(actual_temp_array, forecasted_temp_array)

mae_humidity_model = mean_absolute_error(actual_humidity_array, forecasted_humidity_array)
rmse_humidity_model = root_mean_squared_error(actual_humidity_array, forecasted_humidity_array)
mae_temp_naive = mean_absolute_error(actual_temp_array, naive_temp_array)
rmse_temp_naive = root_mean_squared_error(actual_temp_array, naive_temp_array)

mae_humidity_naive = mean_absolute_error(actual_humidity_array, naive_humidity_array)
rmse_humidity_naive = root_mean_squared_error(actual_humidity_array, naive_humidity_array)
print("Temperature Forecast Error Metrics:")
print(f"Model MAE: {mae_temp_model:.2f}, Model RMSE: {rmse_temp_model:.2f}")
print(f"Naive MAE: {mae_temp_naive:.2f}, Naive RMSE: {rmse_temp_naive:.2f}")

print("\nHumidity Forecast Error Metrics:")
print(f"Model MAE: {mae_humidity_model:.2f}, Model RMSE: {rmse_humidity_model:.2f}")
print(f"Naive MAE: {mae_humidity_naive:.2f}, Naive RMSE: {rmse_humidity_naive:.2f}")
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(test_df.index, actual_temp_array, label='Actual Temperature', color='black')
plt.plot(test_df.index, forecasted_temp_array, label='Model Forecast', color='blue')
plt.plot(test_df.index, naive_temp_array, label='Naive Forecast', color='red', linestyle='--')
plt.title("Temperature Forecast every 15 minutes Comparison")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(test_df.index, actual_humidity_array, label='Actual Humidity', color='black')
plt.plot(test_df.index, forecasted_humidity_array, label='Model Forecast', color='green')
plt.plot(test_df.index, naive_humidity_array, label='Naive Forecast', color='orange', linestyle='--')
plt.title("Humidity Forecast every 15 minutes Comparison")
plt.xlabel("Time")
plt.ylabel("Humidity")
plt.legend()

plt.tight_layout()
plt.savefig('Forecast.png')
plt.show()

# naive_forecast_temp = []
# naive_forecast_humidity = []

# naive =  df[(df.index >= intervals[3]) & (df.index < intervals[4])]

# last_temp = naive['Temperature'].iloc[0]
# last_humidity = naive['Humidity'].iloc[0]

# for idx in naive.index:
#     naive_forecast_temp.append(last_temp)
#     naive_forecast_humidity.append(last_humidity)
#     last_temp = naive['Temperature'].loc[idx]
#     last_humidity = naive['Humidity'].loc[idx]
#adf_test(df['Temperature'], 'Temperature')
#adf_test(df['Humidity'], 'Humidity')
#print(df)
# fig, ax = plt.subplots(2, 1, figsize=(12,8))

# plot_acf(df['Temperature'], ax=ax[0], lags=100)
# ax[0].set_title('ACF - Temperature')

# plot_pacf(df['Temperature'], ax=ax[1], lags=100)
# ax[1].set_title('PACF - Temperature')

# plt.tight_layout()
# plt.savefig('PACF-ACF_Temperature.png')

# # Plot ACF and PACF for Humidity
# fig, ax = plt.subplots(2, 1, figsize=(12,8))

# plot_acf(df['Humidity'], ax=ax[0], lags=100)
# ax[0].set_title('ACF - Humidity')

# plot_pacf(df['Humidity'], ax=ax[1], lags=100)
# ax[1].set_title('PACF - Humidity')

# plt.tight_layout()
# plt.savefig('PACF-ACF_Humidity.png')

# df['interval'] = pd.cut(df.index, bins=intervals, labels=intervals[:-1])
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
# plt.xlim(0, 24)
# plt.grid(True)
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
