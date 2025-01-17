import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt

# 读取数据
bitcoin = pd.read_excel("data/BCHAIN-MKPRU-new.xlsx")
gold = pd.read_excel("data/LBMA-GOLD.xlsx")

bit_value = bitcoin["Value"].values
bit_date = bitcoin["Date"].values

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(bit_value.reshape(-1, 1))

# 创建时间序列数据集
look_back = 60
generator = TimeseriesGenerator(scaled_data, scaled_data, length=look_back, batch_size=1)

# 分割数据集
X, y = generator[0]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 构建 LSTM 模型
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(X, y, epochs=100, verbose=0)

# 预测
predictions = model.predict(X)
predictions = scaler.inverse_transform(predictions)

# 计算误差
error = np.sqrt(np.mean((predictions.flatten() - bit_value[look_back:]) ** 2))

# 绘制结果
plt.figure(figsize=(12, 6))
plt.plot(bit_date[look_back:], bit_value[look_back:], label='Actual')
plt.plot(bit_date[look_back:], predictions, color='red', label='Predicted')
plt.fill_between(bit_date[look_back:], predictions - 1.96 * np.std(predictions), predictions + 1.96 * np.std(predictions), color='pink', alpha=0.3, label='95% CI')
plt.title('Bitcoin Time Series Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# 保存预测结果
value_pred = predictions.flatten()
error_val = [error] * len(value_pred)
var_numpy = np.concatenate((bit_date[look_back:].reshape((-1, 1)), value_pred.reshape((-1, 1)), error_val.reshape((-1, 1))), axis=1)
df = pd.DataFrame(var_numpy, columns=['Date', 'Prediction', 'Error'])
df.to_excel('bitcoin-solve-neural-network.xlsx', index=False)
