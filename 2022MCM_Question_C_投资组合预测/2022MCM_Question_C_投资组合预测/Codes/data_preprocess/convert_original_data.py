import numpy as np
import pandas as pd

bit = pd.read_excel(r"C:\Users\Lenovo\Desktop\sorts\LBMA-GOLD.xlsx")


def tes(word):
    if type(word) == str:
        return word.split('/')
    else:
        return str(word)[2: 10].split("-")



def to_num(word_list):
    for num in range(len(word_list)):
        try:
            word_list[num] = eval(word_list[num])
        except SyntaxError:
            word_list[num] = eval(word_list[num][1:])
    return word_list


date = bit["Date"].to_numpy()
value = bit["USD (PM)"].to_numpy()

vars = []
for i in range(len(date)):
    if tes(date[i]):
        vars.append([to_num(tes(date[i])), value[i]])

a = [[[1, 2, 3], 5], [[3, 5, 2], 6]]

def sort_data(var):
    var.sort(key=lambda x: x[0][1])
    var.sort(key=lambda x: x[0][0])
    var.sort(key=lambda x: x[0][2])
    return var

sort_data(vars)

def convert_list(data):
    return "20" + str(data[2]) + "/" + str(data[0]) + "/" + str(data[1])

for i in range(len(vars)):
    vars[i][0] = convert_list(vars[i][0])
var_numpy = np.array(vars)

print(var_numpy)
df = pd.DataFrame(var_numpy, columns=['Date', 'USD (PM)'])
df.to_excel('LBMA-GOLD-new.xlsx', index=False)

