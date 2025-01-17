import numpy as np

def encode(start, end, nums):
    if start == end:
        def func_encode(x):
            return np.array([0 for _ in range(nums)])
    else:
        k = (2 ** nums - 1) / (end - start)
        b = - k * start

        def func_encode(x):
            y = int(b + k * x)
            counter = nums - 1
            ls = []
            while counter >= 0:
                # print(y)
                ls.append(y // 2 ** counter)
                y -= (y // 2 ** counter) * 2 ** counter
                counter -= 1
            return np.array(ls)

    return func_encode


def decode(start, end, encode_ls, origin_x):
    if start == end:
        return origin_x
    else:
        slope = (2 ** len(encode_ls) - 1) / (end - start)
        b = - slope * start
        nums = 0
        decode_y = 0
        while nums <= len(encode_ls)-1:
            decode_y += encode_ls[nums] * 2 ** (-nums + len(encode_ls) - 1)
            nums += 1
        decode_x = (decode_y - b) / slope
        return decode_x


def variation(param, probability=0.1):
    a = np.random.random()
    for b in range(len(param)):
        if a <= probability:
            if param[b] == 1:
                param[b] = 0
            else:
                param[b] = 1
    return param


def exchange_outer(param_matrix, stick=None, probability=0.1):
    a = np.random.random()
    best_matrix = np.array(param_matrix[stick, :])
    [m, n] = np.random.randint(0, param_matrix.shape[1], size=2)
    c = np.random.randint(0, param_matrix.shape[0])
    d = np.random.randint(0, best_matrix.shape[0])
    while c in stick:
        c = np.random.randint(0, param_matrix.shape[0])
        # print("Exchange_outer: Notice! c or d are unlucky to contain in stick_list, now have to choose again!")
    if a <= probability:
        temp = best_matrix[d][m:n]
        param_matrix[c][m:n] = temp
    return param_matrix


def exchange_inner(param, probability=0.1):
    a = np.random.random()
    [m, n] = np.random.randint(0, param.shape[0], size=2)
    if a <= probability:
        temp = param[m]
        param[m] = param[n]
        param[n] = temp
    return param


def select(probabilities, scaler=20):
    for i in range(len(probabilities)):
        probabilities[i] = min(1, scaler * probabilities[i])
    return probabilities


def GA(x_list, num, stick=None, prob_variation=0.01, prob_mutation=0.01):
    if num not in stick:
        x_list[i] = variation(x_list[i], prob_variation)
        x_list[i] = exchange_inner(x_list[i], prob_mutation)
    return x_list
