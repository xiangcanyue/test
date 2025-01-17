import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

gold = pd.read_excel("data/LBMA-GOLD-new.xlsx")
gold_value = gold["USD (PM)"][60:]
date = gold["Date"][60:].to_numpy()

pred_val = []
error_val = []
val_low = []
val_high = []

num = 60
for file in date:
    with open("data/log-gold/" + file.replace("/", "-") + ".log") as fo:
        print(file.replace("/", "-") + ".log")
        count = 0
        for line in fo:
            if line.startswith("Interval"):
                count = 1
                continue
            if count == 1:
                num_pred = line.split("\t")
                break
        for i in range(2):
            num_pred[i] = eval(num_pred[i])
        pred_value = sum(num_pred) / 2
        error_value = np.sqrt(np.sum((pred_value - gold_value[num]) ** 2))
        pred_val.append(pred_value)
        error_val.append(error_value)
        val_low.append(num_pred[0])
        val_high.append(num_pred[1])
        num += 1

# date = date.reshape((-1, 1))
# value_pred_numpy = np.array(pred_val).reshape((-1, 1))
# error_val_numpy = np.array(error_val).reshape((-1, 1))
# # value_pred = np.zeros_like(bit_date).reshape((-1, 1))
# # error_val = np.zeros_like(bit_date).reshape((-1, 1))
# var_numpy = np.concatenate((date, value_pred_numpy, error_val_numpy), axis=1)
# df = pd.DataFrame(var_numpy, columns=['Date', 'Prediction', 'Error'])
# df.to_excel('bitcoin-solve-all.xlsx', index=False)

pred_val = np.array(pred_val)
error_val = np.array(error_val)
plt.clf()
plt.figure(figsize=(10, 6))
plt.plot(date, gold_value, label='Original', color='black', linewidth=1)
plt.plot(date, pred_val, label='Forecast', color='green', linestyle='--', linewidth=1)
plt.fill_between(date, val_low, val_high, color='red', alpha=0.3, label='95% CI')
plt.title('Gold Time Series Forecast with 95% Confidence Interval')
plt.xlabel('Date')
plt.ylabel('USD (PM)')
plt.legend()
plt.xticks([date[0], date[-1]])
plt.show()
