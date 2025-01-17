import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import os


bitcoin = pd.read_excel("data/BCHAIN-MKPRU-new.xlsx")
gold = pd.read_excel("data/LBMA-GOLD.xlsx")

bit_date = bitcoin["Date"].to_numpy()
bit_value = bitcoin["Value"].to_numpy()


fore_list = []
error_val_list = []
abs_error_list = []
for i in range(7, 60):
    print(bitcoin["Date"][i])
    df = pd.DataFrame({
        'ds': bit_date[: i],
        'y': bit_value[: i]
        })

    # df = df.rename(columns={'date_column': 'ds', 'value_column': 'y'})

    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
    forecast['yhat_lower'].to_numpy()
    fore_list.append(forecast['yhat'][0])
    error_val = np.abs(forecast['yhat'][0] - bit_value[i])
    error_val_list.append(error_val)
    abs_error = error_val / bit_value[i]
    abs_error_list.append(abs_error)


    plt.clf()
    plt.plot(forecast['ds'][0:i], bit_value[:i], label='Original')
    plt.plot(forecast['ds'][0:i], forecast['yhat'][0:i], label='Forecast', color='red')
    plt.plot(forecast['ds'][i:i+min(len(bit_value), i+7)-i],
             bit_value[i: min(len(bit_value), i+7)], label='Actual', color='green')
    plt.plot(forecast['ds'][i+1:i+8], forecast['yhat'][i+1:i+8], label='Forecast', color='red')
    plt.fill_between(forecast['ds'][i+1:i+8], forecast['yhat_lower'][i+1:i+8].to_numpy().astype('float32'), forecast['yhat_upper'][i+1:i+8].to_numpy().astype('float32'), alpha=0.2, color='red', label='95%CI')
    plt.title(bitcoin["Date"][i].replace('/', '-') + '  Bitcoin Prophet Model Forecast with 95% Confidence Interval')
    plt.legend()
    os.makedirs("data/Bitcoin-Prophet", exist_ok=True)
    os.makedirs("data/Bitcoin-Prophet/predictions", exist_ok=True)
    plt.savefig("data/Bitcoin-Prophet/predictions/" + bitcoin["Date"][i].replace('/', '-') + "_Prophet_prediction.png")
    # plt.show()

fore_list_numpy = np.array(fore_list).reshape((-1, 1))
bit_date_numpy = bitcoin["Date"][7:60].to_numpy().reshape((-1, 1))
error_val_numpy = np.array(error_val_list).reshape((-1, 1))
abs_error_numpy = np.array(abs_error_list).reshape((-1, 1))
var_numpy = np.concatenate((bit_date_numpy, fore_list_numpy, error_val_numpy, abs_error_numpy), axis=1)
df = pd.DataFrame(var_numpy, columns=['Date', 'Prediction', 'Error', 'Relative Error'])
df.to_excel('Bitcoin-Prophet-0.xlsx', index=False)
