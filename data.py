# -*- coding:utf-8 -*-
"""
----------------------------------------------------------
@function:
@time:2024-10-24 20:42:43
@author:zhangbaoquan01@cnpc.com.cn
----------------------------------------------------------
"""
import random
import numpy as np
import torch

vocabStr = '<SOS>,<EOS>,<PAD>,0,1,2,3,4,5,6,7,8,9,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z'
xDict = {word: index for index, word in enumerate(vocabStr.split(','))}
xCharList = [word for word, index in xDict.items()]
yDict = {word.upper():index for word, index in xDict.items()}
yCharList = [word for word, index in yDict.items()]

def getOneData():
    """
    生成一条训练数据样本
    :return:
    """
    def mappingFun(char):
        char = char.upper()
        if not char.isdigit():
            return char
        char = 9 - int(char)
        return str(char)
    # 定义词集合
    words = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'q', 'w', 'e', 'r',
             't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k',
             'l', 'z', 'x', 'c', 'v', 'b', 'n', 'm']
    # 定义每个词先中的概率
    percent = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
             13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
    percent = percent / sum(percent)
    # 一句话随机的字符个数
    num = random.randint(30, 47)
    # 生成一句话
    x = np.random.choice(words, size=num, replace=True, p=percent)
    x = x.tolist()
    # 根据映射函数生成
    y = [mappingFun(char) for char in x]
    y = y + [y[-1]]
    y = y[::-1]
    # 加上首尾的开始、结束标识符号
    x = ['<SOS>'] + x + ['<EOS>']
    y = ['<SOS>'] + y + ['<EOS>']
    # 将句子填充pad字符
    x = x + ['<PAD>'] * 50
    y = y + ['<PAD>'] * 51
    # 将句子截断到相同的字符
    x = x[:50]
    y = y[:51]
    # 将明文的句子映射成编码数据
    x = [xDict[i] for i in x]
    y = [yDict[i] for i in y]
    # 转成张量
    x = torch.LongTensor(x)
    y = torch.LongTensor(y)
    return x, y

class DataSet(torch.utils.data.Dataset):

    def __init__(self):
        super(DataSet, self).__init__()

    def __len__(self):
        return 100000

    def __getitem__(self, i):
        return getOneData()

loader = torch.utils.data.DataLoader(dataset=DataSet(), batch_size=8, shuffle=True, drop_last=True, collate_fn=None)

