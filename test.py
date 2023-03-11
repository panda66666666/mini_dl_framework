import sys
from textwrap import indent

sys.path.append('.')
import random
import visdom
import Pypanda
import numpy as np
import paddle.fluid as fluid
import paddle
import paddle.vision.transforms as T
import os
import matplotlib.pyplot as plt

vis = visdom.Visdom(env='plot1')
''' demo0:预测初等函数'''
#m=Pypanda.map()

#x = Pypanda.Variable(0)
##将输入加入到数据流图中
#x >> m

##创建需要预测的参数
#a = Pypanda.Variable(0)
#m.AddParaNode(a.py_node)
#b = Pypanda.Variable(0)
#m.AddParaNode(b.py_node)
#c = Pypanda.Variable(0)
#m.AddParaNode(c.py_node)
#d = Pypanda.Variable(0)
#m.AddParaNode(d.py_node)

##创建标签量
#label = Pypanda.Variable(0)

##定义数据流图
#y = a*x**2+b*x+c

##定义损失函数
#loss = Pypanda.MSELoss(label, y)

##将损失函数加入到数据流图中
#m << loss

#time = 0
#while time < 100000:
#    x_ = random.randint(1, 100)
#    x.SetData(x_)
#    label.SetData(3*x_**2+2*x_+1)

#    #数据流图正向传播
#    m.Forward()
#    #数据节点梯度归零
#    m.SetGradZero()
#    #数据流图反向传播
#    m.BackWard()
#    m.UpdatePara(eta=0.00000000001)

#    x_plot = np.arange(-10, 10, 0.1)
#    y_plot = a.GetData()*x_plot**2+b.GetData()*x_plot+c.GetData()

#    y_ori_plot = 3*x_plot**2+2*x_plot+1

#    vis.line(X=x_plot, Y=np.column_stack((y_ori_plot, y_plot)), win='3.x')

#    print('a= ', a.GetData(), '  ', 'b= ', b.GetData(), ' ', 'c= ',
#        c.GetData())

#    time += 1

# useless = 1
'''demo1:波士顿房价预测'''
#BUF_SIZE=500
#BATCH_SIZE=1

##用于训练的数据提供器，每次从缓存中随机读取批次大小的数据
#train_reader = paddle.batch(
#    paddle.reader.shuffle(paddle.dataset.uci_housing.train(),
#                          buf_size=BUF_SIZE),
#    batch_size=BATCH_SIZE)
##用于测试的数据提供器，每次从缓存中随机读取批次大小的数据
#test_reader = paddle.batch(
#    paddle.reader.shuffle(paddle.dataset.uci_housing.test(),
#                          buf_size=BUF_SIZE),
#    batch_size=BATCH_SIZE)

#train_data=paddle.dataset.uci_housing.train()
#test_data=paddle.dataset.uci_housing.test()

##声明更新节点和网路层
#m=Pypanda.map()
#linear1=Pypanda.Linear(13,20)
#linear2=Pypanda.Linear(20,10)
#linear3=Pypanda.Linear(10,1)
#x_in=Pypanda.Variable(np.ndarray(0))
#label=Pypanda.Variable(np.ndarray(0))

##定义数据流图
#x_in>>m
#x=linear1(x_in)
#x=Pypanda.ReLU(x)
#x=linear2(x)
#x=Pypanda.ReLU(x)
#x=linear3(x)
#loss=Pypanda.MSELoss(x,label)
#m<<loss

#times=0
#my_win=None
#my_win_test=None
#while times<10000:
#    data_t=next(train_data())

#    x_in.SetData(np.array(data_t[0])[np.newaxis,:])
#    label.SetData(np.array(data_t[1])[np.newaxis,:])

#    m.SetGradZero()
#    m.Forward()
#    m.Backward()
#    m.UpdatePara(0.001)

#    x_plot=times
#    y_plot=loss.GetData()

#    if times==0:
#       my_win = vis.line(X=np.array([x_plot]),Y=y_plot[np.newaxis], opts=dict(title='train_loss'))

#    else:
#        vis.line(X=np.array([x_plot]),Y=y_plot[np.newaxis],win=my_win,update='append')

#    if times%10==0:
#        data_test=next(test_data())

#        x_in.SetData(np.array(data_test[0])[np.newaxis,:])
#        label.SetData(np.array(data_test[1])[np.newaxis,:])

#        m.Forward()
#        x_plot=times
#        y_plot=loss.GetData()

#        if times==0:
#            my_win_test = vis.line(X=np.array([x_plot]),Y=y_plot[np.newaxis], opts=dict(title='test_loss'))

#        else:
#            vis.line(X=np.array([x_plot]),Y=y_plot[np.newaxis],win=my_win_test,update='append')

#    print(loss.GetData())
#    times+=1
'''demo2:手写数字识别'''
# EPOCH=10
# LR=0.01
# def one_hot(num):
#     index=int(num[0])
#     res=np.zeros([1,10])
#     res[0][index]=1
#     return res

# transform = T.Normalize(mean=[127.5], std=[127.5])
# #训练数据集
# train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
# #评估数据集
# eval_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)
# #print('训练集样本量： {}, 验证集样本量： {}'.format(len(train_dataset), len(eval_dataset)))

# #声明更新节点和网路层
# m=Pypanda.map()
# conv2d_1=Pypanda.Conv2d(1,1,3)
# conv2d_2=Pypanda.Conv2d(1,1,3)
# avgpool_1=Pypanda.AvgPool2d(2,2)
# avgpool_2=Pypanda.AvgPool2d(2,2)
# linear=Pypanda.Linear(25,10)

# x_in=Pypanda.Variable(np.ndarray(0))
# label=Pypanda.Variable(np.ndarray(0))

# #定义数据流图
# x_in>>m

# x=conv2d_1(x_in)
# x=avgpool_1(x)
# x=conv2d_2(x)
# x=avgpool_2(x)
# x=Pypanda.reshape(x,[1,-1])
# x=linear(x)
# x=Pypanda.sigmoid(x)

# loss=Pypanda.CELoss(x,label)

# m<<loss

# #x_in.SetData((train_dataset[0][0])[np.newaxis,:])
# #label.SetData(one_hot(train_dataset[0][1]))

# my_win_test=None
# e=0
# while e<EPOCH:
#     ran=random.sample(range(0,len(train_dataset)),len(train_dataset))
#     for number,i in enumerate(ran):
#         x_in.SetData((train_dataset[i][0])[np.newaxis,:])
#         label.SetData(one_hot(train_dataset[i][1]))

#         m.SetGradZero()
#         m.Forward()
#         m.Backward()
#         m.UpdatePara(0.01)

#         x_plot=number+len(train_dataset)*e

#         y_plot=loss.GetData()

#         if number%3001==3000:
#             LR/=10

#         if number==0 and e==0:
#             my_win_test=vis.line(X=np.array([x_plot]),Y=y_plot[np.newaxis], opts=dict(title='train_loss'))
#         else:
#             vis.line(X=np.array([x_plot]),Y=y_plot[np.newaxis],win=my_win_test,update='append')

#     e+=1
