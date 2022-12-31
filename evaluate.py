import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt


'''模型载入'''
model_dir = 'model.pkl'
testdata_dir = 'data\DATA_testing_studentID.csv'
input_cols = ['n1', 'n2', 'T2', 'P2', 'T3', 'P3', 'T34', 'P34', 'T4', 'gc1', 'Qmfs']

'''数据格式转换'''
net = torch.load(model_dir)
net.eval()
data = pd.read_csv(testdata_dir)
Len = len(data)
data_in = data[input_cols]
data_in = (data_in - data_in.min()) / (data_in.max()-data_in.min())
input = data_in.values
input = torch.from_numpy(input).float()

'''类型判断'''
with torch.no_grad():
    out = net(input)
    pred = pd.DataFrame(np.round(out.data.numpy()))
    # pred = np.round(out.data.numpy())
    data['type'] = pred
data.to_csv(testdata_dir)

'''
X = [0, 1, 2, 3, 4, 5, 6]
Y = [0, 0, 0, 0, 0, 0, 0]
for i in range(0, 356):
    for j in range(0, 7):
        if pred[i] == j:
            Y[j] += 1

plt.bar(X, Y, 0.4, color="Blue")
plt.xlabel(u"故障类型", fontproperties='SimHei', fontsize=14)
plt.ylabel(u"数目", fontproperties='SimHei', fontsize=14)

plt.show()
'''




