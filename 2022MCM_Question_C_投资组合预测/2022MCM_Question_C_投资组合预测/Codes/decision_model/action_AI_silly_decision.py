import pandas as pd
import numpy as np

# 假设以下是你的数据，这里只是示例数据，你需要替换为真实数据
date_range = pd.date_range(start='2016-09-11', end='2021-09-10')
bitcoin = pd.read_excel("BCHAIN-MKPRU-new.xlsx")
gold = pd.read_excel("LBMA-GOLD.xlsx")
bitcoin_prices = bitcoin["Value"]
# gold_prices = pd.Series(np.random.rand(len(date_range)) * 2000, index=date_range)

# 初始资金
initial_cash = 1000
cash = initial_cash
bitcoin_quantity = 0
gold_quantity = 0

# 存储每个交易日的资产价值
portfolio_value = []

for i in range(1, len(date_range)):
    # 计算资产的当前价值
    current_value = (cash +
                     bitcoin_quantity * bitcoin_prices.iloc[i]) #+
                     # gold_quantity * gold_prices.iloc[i])
    portfolio_value.append(current_value)

    # 简单的均值回归策略示例
    # 如果价格低于过去一段时间的平均值，则买入；如果高于平均值，则卖出
    bitcoin_mean = bitcoin_prices.iloc[:i].mean()
    # gold_mean = gold_prices.iloc[:i].mean()

    if bitcoin_prices.iloc[i] < bitcoin_mean:
        # 假设我们用 10% 的现金买入比特币
        buy_bitcoin = 2 * cash / bitcoin_prices.iloc[i]
        bitcoin_quantity += buy_bitcoin
        cash -= buy_bitcoin * bitcoin_prices.iloc[i]
    elif bitcoin_prices.iloc[i] > bitcoin_mean:
        # 假设我们卖出 10% 的比特币
        sell_bitcoin = 2 * bitcoin_quantity
        bitcoin_quantity -= sell_bitcoin
        cash += sell_bitcoin * bitcoin_prices.iloc[i]

    # if gold_prices.iloc[i] < gold_mean:
    #     # 假设我们用 10% 的现金买入黄金
    #     buy_gold = 0.1 * cash / gold_prices.iloc[i]
    #     gold_quantity += buy_gold
    #     cash -= buy_gold * gold_prices.iloc[i]
    # elif gold_prices.iloc[i] > gold_mean:
    #     # 假设我们卖出 10% 的黄金
    #     sell_gold = 0.1 * gold_quantity
    #     gold_quantity -= sell_gold
    #     cash += sell_gold * gold_prices.iloc[i]

# 计算最终资产价值
final_value = (cash +
               bitcoin_quantity * bitcoin_prices.iloc[-1])# +
               # gold_quantity * gold_prices.iloc[-1])

print(f"初始资金: {initial_cash}")
print(f"最终资金: {final_value}")
print(f"利润: {final_value - initial_cash}")

# 将投资组合价值存储为 DataFrame
portfolio_df = pd.DataFrame({
    'Portfolio Value': portfolio_value
}, index=date_range[1:])

print(portfolio_df)
