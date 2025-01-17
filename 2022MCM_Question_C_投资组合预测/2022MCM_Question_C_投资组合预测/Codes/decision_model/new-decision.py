import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms

bitcoin_original = pd.read_excel("data/BCHAIN-MKPRU-new.xlsx")
bitcoin_early = pd.read_excel("data/bitcoin-solve-0.xlsx")
bitcoin_prediction = pd.read_excel("data/Bitcoin-solve-all.xlsx")
gold_original = pd.read_excel("data/LBMA-GOLD-new.xlsx")
gold_early = pd.read_excel("data/gold-solve-0.xlsx")
gold_predictions = pd.read_excel("data/Gold-solve.xlsx")

date_list = bitcoin_original["Date"].to_numpy()
gold_date_list = gold_original["Date"].to_numpy()

bit_original = bitcoin_original["Value"].to_numpy()
bit_early_prediction = bitcoin_early["Prediction"].to_numpy()
bit_prediction = bitcoin_prediction["Prediction"].to_numpy()
gold_original = gold_original["USD (PM)"].to_numpy()
gold_early_prediction = gold_early["Prediction"].to_numpy()
gold_prediction = gold_predictions["Prediction"].to_numpy()

# Initialize the items
past_avg_bit_list = []
past_avg_gold_list = []
rate_bit_list = []
rate_gold_list = []
s1_bit_list = []
s2_bit_list = []
s3_bit_list = []
s1_gold_list = []
s2_gold_list = []
s3_gold_list = []

# Calculate s of bitcoin
total_s1 = 0
count = 0

for num in range(7, len(date_list)):
    date = date_list[num]
    if num < 67:
        bit_val = bit_original[7:num + 1]
        bit_pred = bit_prediction[0:num - 6]
        cof = (bit_pred[1:] - bit_pred[:-1]) * (bit_val[1:] - bit_val[:-1]) + 1e-5 if num != 7 else 0
        past_avg_bit = np.average(bit_original[0:num])
        rate_bit = (bit_prediction[num - 6] - bit_original[num]) / bit_original[num] if num != len(date_list) - 1 else 0
        s3_bit = np.sum((bit_val - np.average(bit_val)) ** 2) / len(bit_val)
        s2_bit = np.sum((bit_pred - bit_val) / bit_val) / len(bit_val)
        s1_bit = np.sum(cof / np.abs(cof) + 1) / len(cof) / 2 if num != 7 else 0
        total_s1 += s1_bit
        count += 1
    if num >= 67:
        bit_val = bit_original[num - 60: num + 1]
        bit_pred = bit_prediction[num - 67:num - 6]
        cof = (bit_pred[1:] - bit_pred[:-1]) * (bit_val[1:] - bit_val[:-1]) + 1e-5
        past_avg_bit = np.average(bit_original[num - 60:num])
        rate_bit = (bit_prediction[num - 6] - bit_original[num]) / bit_original[num] if num != len(date_list) - 1 else 0
        s3_bit = np.sum((bit_val - np.average(bit_val)) ** 2) / len(bit_val)
        s2_bit = np.sum((bit_pred - bit_val) / bit_val) / len(bit_val)
        s1_bit = np.sum(cof / np.abs(cof) + 1) / len(cof) / 2
        total_s1 += s1_bit
        count += 1

    past_avg_bit_list.append(past_avg_bit)
    rate_bit_list.append(rate_bit)
    s1_bit_list.append(s1_bit)
    s2_bit_list.append(s2_bit)
    s3_bit_list.append(s3_bit)

s1_avg = total_s1 / count
print(s1_avg)

# Calculate s of gold
total_s1_gold = 0
count = 0
for num in range(7, len(gold_date_list)):
    date = gold_date_list[num]
    # print(date)
    if num < 67:
        gold_val = gold_original[7:num + 1]
        gold_pred = gold_prediction[0:num - 6]
        cof = (gold_pred[1:] - gold_pred[:-1]) * (gold_val[1:] - gold_val[:-1]) + 1e-5 if num != 7 else 0
        s1_gold = np.sum(cof / np.abs(cof) + 1) / len(cof) / 2 if num != 7 else 0
        s2_gold = np.sum((gold_pred - gold_val) / gold_val) / len(gold_val)
        s3_gold = np.sum((gold_val - np.average(gold_val)) ** 2) / len(gold_val)
        rate_gold = (gold_prediction[num - 6] - gold_original[num]) / gold_original[num] if num != len(
            gold_date_list) - 1 else 0
        past_avg_gold = np.average(gold_original[0:num])
        total_s1_gold += s1_gold
        # print(s1_gold)
        count += 1
    if num >= 67:
        gold_val = gold_original[num - 60: num + 1]
        gold_pred = gold_prediction[num - 67:num - 6]
        cof = (gold_pred[1:] - gold_pred[:-1]) * (gold_val[1:] - gold_val[:-1]) + 1e-5
        s1_gold = np.sum(cof / np.abs(cof) + 1) / len(cof) / 2
        s2_gold = np.sum((gold_pred - gold_val) / gold_val) / len(gold_val)
        s3_gold = np.sum((gold_val - np.average(gold_val)) ** 2) / len(gold_val)
        rate_gold = (gold_prediction[num - 6] - gold_original[num]) / gold_original[num] if num != len(
            gold_date_list) - 1 else 0
        past_avg_gold = np.average(gold_original[num - 60:num])
        total_s1_gold += s1_gold
        # print(s1_gold)
        count += 1

    past_avg_gold_list.append(past_avg_gold)
    rate_gold_list.append(rate_gold)
    s1_gold_list.append(s1_gold)
    s2_gold_list.append(s2_gold)
    s3_gold_list.append(s3_gold)

s1_avg = total_s1 / count
print(s1_avg)


def normalize(data):
    data = np.array(data)
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)


s1_norm_bit = normalize(s1_bit_list)
s2_norm_bit = normalize(s2_bit_list)
s3_norm_bit = normalize(s3_bit_list)
s1_norm_gold = normalize(s1_gold_list)
s2_norm_gold = normalize(s2_gold_list)
s3_norm_gold = normalize(s3_gold_list)


# Make decisions
alpha_1 = 0
alpha_2 = 0
alpha_3 = 0.05
beta_1 = 0.1
beta_2 = 0.05
beta_3 = 0

alpha_bit_sell = 1e-5
alpha_bit_buy = 1e5
alpha_gold_sell = 1e2
alpha_gold_buy = 1e0
alpha_min_cash = 0

cash = 10000
bit = 12
gold = 13
alpha_bit = 0.02
alpha_gold = 0.01


cash_list = []
gold_list = []
bitcoin_list = []
final_list = []
count = 0
for num in range(7, len(date_list)):
    date = date_list[num]
    buying_dict = {"bitcoin": 0, "gold": 0}

    if bit_prediction[num - 7] >= past_avg_bit_list[num - 7]:
        # print("bit high")
        # print((bit_prediction[num - 7] - past_avg_bit_list[num - 7]) / past_avg_bit_list[num - 7])
        bit_sell_rate = ((bit_prediction[num - 7] - past_avg_bit_list[num - 7])**2 / past_avg_bit_list[num - 7]
                         * (1 + rate_bit_list[num - 7]) * alpha_bit_sell
                         * (alpha_1 * s1_norm_bit[num - 7] + alpha_2 * (1 - s2_norm_bit[num - 7]) + alpha_3 * (
                            1 - s3_norm_bit[num - 7])))
        # print(bit_sell_rate)
        cash += (1 - alpha_bit) * bit_original[num] * min(bit_sell_rate, 1) * bit
        bit -= min(bit_sell_rate, 1) * bit

    if bit_prediction[num - 7] < past_avg_bit_list[num - 7]:
        # print("bit low")
        bit_buy_rate = ((- bit_prediction[num - 7] + past_avg_bit_list[num - 7])**2 / past_avg_bit_list[num - 7]
                        * (1 - rate_bit_list[num - 7]) * alpha_bit_buy
                        * (alpha_1 * s1_norm_bit[num - 7] + alpha_2 * (1 - s2_norm_bit[num - 7]) + alpha_3 * (
                           1 - s3_norm_bit[num - 7])))
        # print(bit_buy_rate)
        # bit += bit_buy_rate * bit
        # cash -= bit_buy_rate * bit / (1 - alpha_bit) * bit_original[num]  # Notice!!!!!!!!!!
        buying_dict["bitcoin"] = bit_buy_rate

    if date in gold_predictions["Date"].to_numpy():

        if gold_prediction[count] >= past_avg_gold_list[count]:
            gold_sell_rate = ((gold_prediction[count] - past_avg_gold_list[count]) / past_avg_gold_list[count]
                              * (1 + rate_gold_list[count]) * alpha_gold_sell
                              * (beta_1 * s1_norm_gold[count] + beta_2 * (1 - s2_norm_gold[count]) + beta_3 * (
                                 1 - s3_norm_gold[count])))
            # print(gold_sell_rate)
            cash += (1 - alpha_gold) * gold_original[count + 7] * min(gold_sell_rate, 1) * gold
            gold -= min(gold_sell_rate, 1) * gold

        if gold_prediction[count] < past_avg_gold_list[count]:
            gold_buy_rate = ((- gold_prediction[count] + past_avg_gold_list[count]) / past_avg_gold_list[count]
                             * (1 - rate_gold_list[count]) * alpha_gold_buy
                             * (beta_1 * s1_norm_gold[count] + beta_2 * (1 - s2_norm_gold[count]) + beta_3 * (
                                1 - s3_norm_gold[count])))
            # print(gold_buy_rate)
            # gold += gold_buy_rate * gold
            # cash -= gold_buy_rate * gold / (1 - alpha_gold) * gold_original[count+7]  # Notice!!!!!!!!!!
            buying_dict["gold"] = gold_buy_rate

    # Do not use all of your money ！
    if buying_dict["bitcoin"] != 0 or buying_dict["gold"] != 0:
        cash_left = cash - (buying_dict["bitcoin"] + buying_dict["gold"]) * cash
        if cash_left >= alpha_min_cash * cash:
            cash_bit = buying_dict["bitcoin"] * cash
            cash_gold = buying_dict["gold"] * cash
            bit += cash_bit * (1 - alpha_bit) / bit_original[num]
            gold += cash_gold * (1 - alpha_gold) / gold_original[count + 7]
            cash = cash_left
        elif cash_left < alpha_min_cash * cash:
            # Buy golds first, because it is more stable and the prediction accuracy is higher than that of bitcoin.
            # No, buy bitcoin first! You will fly!
            cash_left = cash - buying_dict["bitcoin"] * cash
            if cash_left >= alpha_min_cash * cash:
                cash_bit = buying_dict["bitcoin"] * cash
                bit += cash_bit * (1 - alpha_bit) / bit_original[num]
                # money also left and enough to remain, buy a little gold
                cash_gold = cash_left - cash * alpha_min_cash
                gold += cash_gold * (1 - alpha_gold) / gold_original[count+7]
                cash = alpha_min_cash * cash
            else:
                cash_bit = (1 - alpha_min_cash) * cash
                bit += cash_bit * (1 - alpha_bit) / bit_original[num]
                cash = alpha_min_cash * cash


    cash_list.append(cash)
    gold_list.append(gold)
    bitcoin_list.append(bit)
    final_value = cash + bit * bit_original[num] + gold * gold_original[count + 7]
    final_list.append(final_value)

    print("Date:", date)
    print("Cash:", cash)
    print("Gold:", gold)
    print("Bitcoin", bit)
    print("Final Value:", final_value)
    print("\n")

    if date in gold_predictions["Date"].to_numpy():
        count += 1

date_numpy = date_list[7:].reshape((-1, 1))
gold_numpy = np.array(gold_list).reshape((-1, 1))
bitcoin_numpy = np.array(bitcoin_list).reshape((-1, 1))
cash_numpy = np.array(cash_list).reshape((-1, 1))
final_value_numpy = np.array(final_list).reshape((-1, 1))

var_numpy = np.concatenate((date_numpy, final_value_numpy, cash_numpy, bitcoin_numpy, gold_numpy), axis=1)
df = pd.DataFrame(var_numpy, columns=['Date', 'Total Asset Value', 'Cash', 'Bitcoin', 'Gold'])
df.to_excel('--Decision--.xlsx', index=False)

