import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib import pyplot as plt
import pmdarima as pm
import os


bitcoin = pd.read_excel("data/BCHAIN-MKPRU-new.xlsx")
gold = pd.read_excel("data/LBMA-GOLD.xlsx")

bit_value = bitcoin["Value"]
value_pred = []
error_val = []

# 绘制原始时间序列
# df.plot()
# plt.title('Original Time Series')
# plt.show()



for i in range(7, 60):
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/log-bitcoin", exist_ok=True)
    fo = open("data/log-bitcoin/" + bitcoin["Date"][i].replace('/', '-') + ".log", "w")
    print(bitcoin["Date"][i].replace('/', '-'), file=fo)
    print(bitcoin["Date"][i].replace('/', '-'))

    bit = bit_value[0: i]

    # 使用 auto_arima 寻找最优参数
    model_arima = pm.auto_arima(bit, start_p=0, start_q=0,
                          max_p=10, max_q=10,
                          m=1,  # 季节性周期，对于非季节性数据通常为 1
                          start_P=0, start_Q=0,
                          max_P=10, max_Q=10,
                          seasonal=False,  # 设为 False 表示非季节性模型
                          d=0, D=10, trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)

    print(model_arima.summary(), file=fo)

    # 拟合ARIMA模型
    model = ARIMA(bit, order=model_arima.order)
    model_fit = model.fit()

    # 输出模型摘要
    print(model_fit.summary(), file=fo)
    # 进行预测，包括置信区间
    forecast_result = model_fit.get_forecast(steps=7)
    forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()

    # 输出预测结果
    print(forecast, file=fo)
    value_pred.append(forecast[i+1])
    error = np.sqrt(np.sum((forecast[i+1] - bit_value[i]) ** 2))
    error_val.append(error)

    # 绘制预测结果及其置信区间
    plt.clf()
    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(1, i, i), bit, label='Original')
    plt.plot(np.linspace(i+1, i+7, 7), forecast, label='Forecast', color='red')
    plt.plot(np.linspace(i, i + min(len(bit_value), i+7) - i, min(len(bit_value), i+7) - i + 1), bit_value[i-1: min(len(bit_value), i+7)], label='Actual', color='green')
    plt.fill_between(np.linspace(i+1, i+7, 7), conf_int.iloc[:, 0].to_numpy(), conf_int.iloc[:, 1].to_numpy(), color='pink', alpha=0.3, label='95% CI')
    plt.title(bitcoin["Date"][i].replace('/', '-') + '  Bitcoin Time Series Forecast with 95% Confidence Interval')
    plt.legend()
    os.makedirs("data/Bitcoin-ARIMA", exist_ok=True)
    os.makedirs("data/Bitcoin-ARIMA/predictions", exist_ok=True)
    plt.savefig("data/Bitcoin-ARIMA/predictions/" + bitcoin["Date"][i].replace('/', '-') + "_ARIMA_prediction.png")

    print("Interval:", file=fo)
    for a in range(7):
        print(conf_int.iloc[:, 0].to_numpy()[a], end='\t', file=fo)
        print(conf_int.iloc[:, 1].to_numpy()[a], file=fo)

    # 进行d阶差分
    d = model_arima.order[1]
    df_diff = bit.diff()
    for _ in range(d - 1):
        df_diff = df_diff.diff()

    if np.isinf(df_diff).any() or df_diff.isna().any():
        # 将 inf 替换为 NaN
        df_diff[np.isinf(df_diff)] = np.nan
        # 可以选择删除包含 NaN 的行
        df_diff = df_diff.dropna()
        # 或者填充 NaN 为某个具体的值，例如 0
        # df_diff = df_diff.fillna(0)

    # 绘制差分后的序列图
    plt.clf()
    df_diff.plot()
    plt.title(bitcoin["Date"][i].replace('/', '-') + 'Differential Time Series')
    os.makedirs("data/Bitcoin-ARIMA/Diff-plot", exist_ok=True)
    plt.savefig("data/Bitcoin-ARIMA/Diff-plot/" + bitcoin["Date"][i].replace('/', '-') + "_ARIMA_Diff.png")

    # 进行平稳性检验
    result = adfuller(df_diff)
    print('ADF Statistic:', result[0], file=fo)
    print('p-value:', result[1], file=fo)
    print('p-value:', result[1])

    # 绘制ACF和PACF图
    plt.clf()
    plot_acf(df_diff, lags=5)
    plt.title(bitcoin["Date"][i].replace('/', '-') + 'ACF' + '  p-value=' + str(result[1]))
    os.makedirs("data/Bitcoin-ARIMA/ACF", exist_ok=True)
    plt.savefig("data/Bitcoin-ARIMA/ACF/" + bitcoin["Date"][i].replace('/', '-') + "_ARIMA_ACF.png")

    plt.clf()
    plot_pacf(df_diff, lags=2)
    plt.title(bitcoin["Date"][i].replace('/', '-') + 'PACF' + '  p-value=' + str(result[1]))
    os.makedirs("data/Bitcoin-ARIMA/PACF", exist_ok=True)
    plt.savefig("data/Bitcoin-ARIMA/PACF/" + bitcoin["Date"][i].replace('/', '-') + "_ARIMA_PACF.png")

    print('\n', file=fo)

bit_date = bitcoin["Date"][7: 60].to_numpy().reshape((-1, 1))
value_pred_numpy = np.array(value_pred).reshape((-1, 1))
error_val_numpy = np.array(error_val).reshape((-1, 1))
# value_pred = np.zeros_like(bit_date).reshape((-1, 1))
# error_val = np.zeros_like(bit_date).reshape((-1, 1))
var_numpy = np.concatenate((bit_date, value_pred_numpy, error_val_numpy), axis=1)
df = pd.DataFrame(var_numpy, columns=['Date', 'Prediction', 'Error'])
df.to_excel('bitcoin-solve.xlsx', index=False)
