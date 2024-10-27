# -*- coding:utf-8 -*-
"""
----------------------------------------------------------
@function:
@time:2024-10-25 21:36:18
@author:zhangbaoquan01@cnpc.com.cn
----------------------------------------------------------
"""
import math
import torch

def attention(Q, K, V, mask):
    # Q, K, V: [batchSize, headNum, sentenceLen, oneHeadDim] : [batchSize, 4, 50, 8]
    # Q * K : [batchSize, 4, 50, 8] * [batchSize, 4, 8, 50] -> [batchSize, 4, 50, 50]
    score = torch.matmul(Q, K.permute(0, 1, 3, 2))
    score = score / 8 ** 0.5
    # 掩码, 将mask中值为 true 的位置置为负无穷大
    score = score.masked_fill_(mask, -float('inf'))
    score = torch.softmax(score, dim=-1)
    # 注意思机制与 V 相乘
    # score : [batchSize, 4, 50, 50] * [batchSize, 4, 50, 8] -> [batchSize, 4, 50, 8]
    score = torch.matmul(score, V)
    # 将多个头的结果拼接在一起
    # score : [batchSize, 4, 50, 8] -> [batchSize, 50, 4, 8] -> [batchSize, 50, 32]
    score = score.permute(0, 2, 1, 3).reshape(-1, 50, 32)
    return score

class EmbeddingWithPosition(torch.nn.Module):

    def __init__(self):
        super(EmbeddingWithPosition, self).__init__()
        rowNum, colNum = 50, 32
        def getPE(rowIndex, colIndex, dModel):
            """
            根据行、列索计算矩阵中每个特定元素的值
            dModel: 词嵌入的维度
            """
            fenmu = 1e4 ** (colIndex / dModel)
            pe = rowIndex / fenmu
            if colIndex % 2 == 0:
                return math.sin(pe)
            return math.cos(pe)
        # 初始化位置编码矩阵
        PE = torch.empty(rowNum, colNum)
        for i in range(rowNum):
            for j in range(colNum):
                PE[i][j] = getPE(i, j, colNum)
        PE = PE.unsqueeze(0)
        # 定义为不更新的常量
        self.register_buffer("PE", PE)
        # 词嵌入层, num_embeddings: 词表大小, embedding_dim: 词嵌入的维度
        self.embed = torch.nn.Embedding(39, colNum)
        # 初始化参数
        self.embed.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # [batchSize, sentenceLen] [8, 50] -> [8, 50, 32]
        embedX = self.embed(x)
        # 词嵌入编码与位置编码相加
        # [8, 50, 32] + [1, 50, 32] -> [8, 50, 32]
        embedX = embedX + self.PE
        return embedX


class MultHeadAttention(torch.nn.Module):

    def __init__(self):
        super(MultHeadAttention, self).__init__()
        # fcQ, fcK, fcV 是头数为4，每个头维度为8的线性层，词嵌入输入与此三个矩阵相乘，得到4个头的拼接输出
        # 此相当于使用4个头为8的线性层相乘输出维度为8，4个头再拼接成32维
        self.fcQ = torch.nn.Linear(32, 32)
        self.fcK = torch.nn.Linear(32, 32)
        self.fcV = torch.nn.Linear(32, 32)
        self.fcOut = torch.nn.Linear(32, 32)
        # 归一化层
        self.norm = torch.nn.LayerNorm(normalized_shape=32, elementwise_affine=True)
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, Q, K, V, mask):
        batchSize = Q.shape[0]
        seqLen = Q.shape[1]
        # x [batchSize, sentenceLen, 32],即 [8, 50, 32]
        # 复制原始 x,用于后面与残差相加
        qSrc = Q.clone()
        # 首先归一化, 仅对最后维进行归一化
        Q = self.fcQ(Q)
        K = self.fcK(K)
        V = self.fcV(V)
        # 形成 Q, K, V 三种矩阵,三个矩阵是4个头的拼接
        # [batchSize, 50, 32] * [32, 32] -> [batchSize, 50, 32]
        Q = self.fcQ(Q)
        K = self.fcK(K)
        V = self.fcV(V)
        # 将相乘的结果拆分成4个头，每个头维度为8
        # [batchSize, 50, 32] -> [batchSize, 50, 4, 8]
        # [batchSize, 50, 32] -> [batchSize, 50, 4, 8] -> [batchSize, 4, 50, 8]
        Q = Q.reshape(batchSize, seqLen, 4, 8).permute(0, 2, 1, 3)
        K = K.reshape(batchSize, seqLen, 4, 8).permute(0, 2, 1, 3)
        V = V.reshape(batchSize, seqLen, 4, 8).permute(0, 2, 1, 3)
        # 计算注意力得分 [batchSize, 4, 50, 8] -> [batchSize, 50, 32]
        score = attention(Q, K, V, mask)
        score = self.dropout(self.fcOut(score))
        # 残差连接
        score = qSrc + score
        return score


class FullConnectionOutput(torch.nn.Module):

    def __init__(self):
        super(FullConnectionOutput, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.Dropout(p=0.1)
        )
        self.norm = torch.nn.LayerNorm(normalized_shape=32, elementwise_affine=True)

    def forward(self, x):
        # 作残差链接输入
        xSrc = x.clone()
        # 归一化
        x = self.norm(x)
        # 全连接层 [batchSize, 50, 32] -> [batchSize, 50, 32]
        out = self.fc(x)
        # 残差链接
        out = out + xSrc
        return out
















