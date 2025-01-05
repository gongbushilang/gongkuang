import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import openpyxl

sample = np.array(pd.read_excel("C:\\Users\\RZC\\Desktop\\error correction\\南水data.xlsx", 0))
floodnum = 5  # 场次洪水数
tnum = [41, 41, 41, 41, 41]  # 各场次洪水时段数
pjie = 3  # 自回归阶数

# 1.构造每场洪水的误差数值对，每一个数值对由前三个时段的误差与当前误差组成
xulie = []  # 存储所有场次洪水的误差序列
for i in range(floodnum):
    temp = np.zeros(((tnum[i] - pjie), pjie + 1))
    error = sample[0:tnum[i], 5 + i * 7]  # 读取每个洪水场次的误差序列
    for j in range(tnum[i] - pjie):  # 构建每个洪水场次的误差数值对，三个前期误差对应一个当前误差
        temp[j, :] = error[j:j + pjie + 1]
    xulie.append(temp)

ecorrection = []  # 存储校正后的误差
r_squared = []  # 存储每次模型计算的R方值
for i in range(floodnum):
    train = xulie[:i] + xulie[i + 1:]
    trainmerge = np.concatenate(train, axis=0)  # 整合成一个数组
    test = xulie[i]
    train_x = trainmerge[:, :pjie]
    train_y = trainmerge[:, pjie:]
    test_x = test[:, :pjie]
    test_y = test[:, pjie:]

    # 2.训练模型得到所需参数：AR的滞后项个数p，和自回归函数的各个系数
    LR_model = LinearRegression()
    LR_model.fit(train_x, train_y)
    predict = LR_model.predict(test_x)

    # 计算R方值
    r2 = LR_model.score(test_x, test_y)
    r_squared.append(r2)
    print(f"Flood {i + 1} R^2: {r2:.4f}")

    ecorrection.append(predict)

# 3.把校正后的误差写入excel
excel_file_path = '南水data.xlsx'
workbook = openpyxl.load_workbook(excel_file_path)
sheet_name = 'Sheet1'
sheet = workbook[sheet_name]

for i in range(floodnum):
    for j in range(len(ecorrection[i])):
        sheet.cell(row=j + 2 + pjie, column=7 + 7 * i, value=ecorrection[i][j].item())
    # 保存修改后的Excel文件
    workbook.save(excel_file_path)
workbook.close()