import numpy as np
import pandas as pd

# 用上下两个值的平均值填充缺失值

def fill_missing_with_avg(series):
    """
    该函数用于将输入序列中的缺失值用上下两个非缺失值的平均值填充
    :param series: 输入的 pandas 序列
    :return: 填充后的序列
    """
    index = series.index
    result = series.copy()
    for i in range(len(series)):
        if np.isnan(series.iloc[i]):
            prev_index = index[i - 1] if i > 0 else None
            next_index = index[i + 1] if i < len(series) - 1 else None
            if prev_index is not None and next_index is not None:
                result.iloc[i] = (series.loc[prev_index] + series.loc[next_index]) / 2
    return result

golds = pd.read_excel("data/LBMA-GOLD-new.xlsx")

gold_value = golds["USD (PM)"]

# 填充缺失值
gold_value = fill_missing_with_avg(gold_value).to_numpy().reshape((-1, 1))
gold_date = golds["Date"].to_numpy().reshape((-1, 1))
var_numpy = np.concatenate((gold_date, gold_value), axis=1)
df = pd.DataFrame(var_numpy, columns=['Date', 'USD (PM)'])
df.to_excel("data/LBMA-GOLD-new.xlsx", index=False)
