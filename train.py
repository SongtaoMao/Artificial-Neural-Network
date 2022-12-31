import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Sampler, Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import Networks
import dataset

train_MSE = []
train_r2 = []
test_MSE = []
test_r2 = []
x = []
net = Networks.MLP(input_dim=11, output_dim=1)
# 定义损失函数
Loss_func = nn.MSELoss()
# 定义优化器
Opt = optim.Adam(net.parameters(), lr=0.001)  # 查阅adam原理

Norm_trans = True
trainset_dir = 'data\DATA_training.csv'
train_set = dataset.Dataset0(trainset_dir, trans=Norm_trans)
# print(train_set)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True)
# print(train_loader)

testset_dir = 'data\DATA_validating.csv'
test_set = dataset.Dataset0(testset_dir, trans=Norm_trans)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)


for i in range(2500):

    '''训练'''
    net.train()
    train_loss = 0
    r2_tr = 0
    ii = 0
    for data, target in train_loader:
        out = net(data)
        loss = Loss_func(out, target)

        Opt.zero_grad()
        loss.backward()
        Opt.step()

        train_loss += loss.item()
        r2 = r2_score(out.detach().numpy(), target.detach().numpy())
        r2_tr += r2
        ii += 1

    r2_tr = r2_tr/ii
    MSE = train_loss/ii
    # writer_tr.add_scalar('r2', r2_tr, i)
    # writer_tr.add_scalar('MSE', r2_tr, i)
    print(i, 'train_mse', MSE, 'train_r2', r2_tr)
    train_MSE.append(MSE)
    train_r2.append(r2_tr)

    '''测试'''
    net.eval()
    test_loss = 0
    r2_te = 0
    iii = 0
    with torch.no_grad():
        iii = 0
        for data0, target0 in test_loader:
            out0 = net(data0)
            loss0 = Loss_func(out0, target0)
            test_loss += loss0.item()
            r2_0 = r2_score(out0.detach().numpy(), target0.detach().numpy())
            r2_te += r2_0
            iii += 1
        MSE_0 = test_loss/iii
        r2_te = r2_te/iii
    print(i, 'test_mse', MSE_0, 'test_r2', r2_0)
    test_MSE.append(MSE_0)
    test_r2.append(r2_te)
    x.append(i)

'''绘图'''
plt.figure(1)

plt.subplot(221)
p1, = plt.plot(x, test_MSE, 'r')
plt.xlabel(u'训练次数', fontproperties='SimHei', fontsize=14)
plt.ylabel(u'test_MSE')

plt.subplot(222)
p2, = plt.plot(x, test_r2, 'r')
plt.xlabel(u'训练次数', fontproperties='SimHei', fontsize=14)
plt.ylabel(u'test_r2')
plt.ylim(0, 1)

plt.subplot(211)
p3, = plt.plot(x, train_MSE, 'b')
plt.xlabel(u'训练次数', fontproperties='SimHei', fontsize=14)
plt.ylabel(u'train_MSE')

plt.subplot(212)
p4, = plt.plot(x, train_r2, 'r')
plt.xlabel(u'训练次数', fontproperties='SimHei', fontsize=14)
plt.ylabel(u'train_r2')
plt.ylim(0, 1)

plt.show()

torch.save(net, 'model.pkl')
