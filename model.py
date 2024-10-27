# -*- coding:utf-8 -*-
"""
----------------------------------------------------------
@function:
@time:2024-10-26 09:44:43
@author:zhangbaoquan01@cnpc.com.cn
----------------------------------------------------------
"""

import torch
from util import MultHeadAttention
from util import EmbeddingWithPosition
from util import FullConnectionOutput
from mask import maskPad
from mask import maskTril

class EncoderLayer(torch.nn.Module):

    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.mh = MultHeadAttention()
        self.fc = FullConnectionOutput()

    def forward(self, x, mask):
        # 多头注意力，维度不变
        # [batchSize, seqLen, d_model], 即: [batchSize, 50, 32] -> [batchSize, 50, 32]
        score = self.mh(x, x, x, mask)
        # 全连接层，维度不变
        # [batchSize, seqLen, d_model], 即: [batchSize, 50, 32]
        out = self.fc(score)
        return out

class Encoder(torch.nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = EncoderLayer()
        self.layer2 = EncoderLayer()
        self.layer3 = EncoderLayer()

    def forward(self, x, mask):
        x = self.layer1(x, mask)
        x = self.layer2(x, mask)
        x = self.layer3(x, mask)
        return x

class DecoderLayer(torch.nn.Module):

    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.mh1 = MultHeadAttention()
        self.mh2 = MultHeadAttention()
        self.fc  = FullConnectionOutput()

    def forward(self, x, y, maskPadX, maskTrilY):
        # 多头注意力，维度不变
        # [batchSize, seqLen, d_model], 即: [batchSize, 50, 32] -> [batchSize, 50, 32]
        y = self.mh1(y, y, y, maskTrilY)
        # 结合 encoder 输入，计算多头注意力
        y = self.mh2(y, x ,x, maskPadX)
        # 全连接层，维度不变
        # [batchSize, seqLen, d_model], 即: [batchSize, 50, 32]
        out = self.fc(y)
        return out

class Decoder(torch.nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = DecoderLayer()
        self.layer2 = DecoderLayer()
        self.layer3 = DecoderLayer()

    def forward(self, x, y, maskPadX, maskTrilY):
        y = self.layer1(x, y, maskPadX, maskTrilY)
        y = self.layer2(x, y, maskPadX, maskTrilY)
        y = self.layer3(x, y, maskPadX, maskTrilY)
        return y


class Transformer(torch.nn.Module):

    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.embedX = EmbeddingWithPosition()
        self.embedY = EmbeddingWithPosition()
        self.fcOut = torch.nn.Linear(32, 39)

    def forward(self, x, y):
        # x,y 形状：[batchSize, seqLen] 即：[batchSize, 50]
        # 掩码
        maskPadX = maskPad(x)
        maskTrilY = maskTril(y)
        # 编码,添加位置信息
        # x = [b, 50] -> [b, 50, 32]
        # y = [b, 50] -> [b, 50, 32]
        embedX = self.embedX(x)
        embedY = self.embedY(y)
        # 编码器计算 [b, 50, 32] -> [b, 50, 32]
        encX = self.encoder(embedX, maskPadX)
        # 解码器层计算 [b, 50, 32],[b, 50, 32] -> [b, 50, 32]
        dec = self.decoder(encX, embedY, maskPadX, maskTrilY)
        # 全连接层，输出维度为 [b, 50, 39]
        out = self.fcOut(dec)
        return out


























