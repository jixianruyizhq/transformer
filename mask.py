# -*- coding:utf-8 -*-
"""
----------------------------------------------------------
@function:
@time:2024-10-26 12:21:38
@author:zhangbaoquan01@cnpc.com.cn
----------------------------------------------------------
"""
import torch
from data import xDict, yDict



def maskPad(data):
    """
    输入 data 是未经embedding的字符索引 [batchSize, sentenceLen]
    data.shape = [b, 50]
    """
    mask = data == xDict['<PAD>']
    # [b, 50] -> [b, 1, 1, 50]
    mask = mask.reshape(-1, 1, 1, 50)
    # 在计算注意力时,是计算50个词和50个词相互之间的注意力,所以是个50*50的矩阵
    # 是pad的列是true,意味着任何词对pad的注意力都是0
    # 但是pad本身对其他词的注意力并不是0, 所以是pad的行不是true

    # 复制 n 次
    mask = mask.expand(-1, 1, 50, 50)
    return mask

def maskTril(data):
    """
        输入 data 是未经embedding的字符索引 [batchSize, sentenceLen]
        data.shape = [b, 50]
        # 50*50的矩阵表示每个词对其他词是否可见
        上三角矩阵,不包括对角线,意味着,对每个词而言,他只能看到他自己,和他之前的词,而看不到之后的词
        [1, 50, 50]
        [[0, 1, 1, 1, 1],
         [0, 0, 1, 1, 1],
         [0, 0, 0, 1, 1],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0]]
    """
    tril = 1 - torch.tril(torch.ones(1, 50, 50, dtype=torch.long))
    # 判断y当中每个词是不是pad,如果是pad则不可见
    mask = data == yDict['<PAD>']
    # 变形+转型,为了之后的计算
    # [b, 50] -> [b, 1, 50]
    mask = mask.unsqueeze(1).long()
    # mask和tril求并集
    mask = mask + tril
    # 转布尔型
    mask = mask > 0
    # 转布尔型,增加一个维度,便于后续的计算
    mask = (mask == 1).unsqueeze(dim=1)
    return mask