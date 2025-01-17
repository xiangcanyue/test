import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

bitcoin = pd.read_excel("BCHAIN-MKPRU-new.xlsx")
gold = pd.read_excel("LBMA-GOLD-new.xlsx")
bit_value = bitcoin["Value"]


num = 50

x = np.linspace(1, num, num)

def A(x_val):
    matrix = np.stack((x_val,
                       # x_val ** 2,
                       np.sqrt(x_val),
                       np.sqrt(abs(x_val - 10)),
                       np.sqrt(abs(x_val - 20)),
                       np.sqrt(abs(x_val - 30)),
                       np.sqrt(abs(x_val - 40)),
                       np.sqrt(abs(x_val - 50)),
                       np.sqrt(abs(x_val - 60)),
                       np.sqrt(abs(x_val - 70)),
                       # np.sin(x_val * np.pi / 60),
                       # np.cos(x_val * np.pi / 60),
                       np.sin(x_val * np.pi / 40),
                       np.cos(x_val * np.pi / 40),
                       np.sin(x_val * np.pi / 30),
                       np.cos(x_val * np.pi / 30),
                       np.sin(x_val * np.pi / 20),
                       np.cos(x_val * np.pi / 20),
                       np.sin(x_val * np.pi / 15),
                       np.cos(x_val * np.pi / 15),
                       np.sin(x_val * np.pi / 10),
                       np.cos(x_val * np.pi / 10),
                       np.sin(x_val * np.pi / 8),
                       np.cos(x_val * np.pi / 8),
                       np.sin(x_val * np.pi / 5),
                       np.cos(x_val * np.pi / 5),
                       np.sin(x_val * np.pi / 4),
                       np.cos(x_val * np.pi / 4),
                       np.sin(x_val * np.pi / 3),
                       np.cos(x_val * np.pi / 3),
                       np.sin(x_val * np.pi / 2),
                       np.cos(x_val * np.pi / 2),
                       )).T
    return matrix

for i in range(num, num + 10):
    bit = bit_value[i - num: i]

    # 使用 numpy 的 linalg.lstsq 函数求解最小二乘问题
    sol_x, residuals, rank, s = np.linalg.lstsq(A(x), bit, rcond=None)

    print("最小二乘解 x:", sol_x)
    print("残差平方和 residuals:", residuals)
    print("矩阵 A 的秩 rank:", rank)
    print("奇异值 s:", s)

    plt.plot(x, bit)
    plt.plot(x, np.matmul(A(x), sol_x))

    x_pred = np.linspace(num, num + 1, 2)
    plt.plot(x_pred, np.matmul(A(x_pred), sol_x))
    plt.plot(x_pred, bit_value[i-1: i+1])

    plt.show()
