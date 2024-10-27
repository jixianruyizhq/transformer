# -*- coding:utf-8 -*-
"""
----------------------------------------------------------
@function:
@time:2024-10-26 21:19:01
@author:zhangbaoquan01@cnpc.com.cn
----------------------------------------------------------
"""
import torch
from data import yDict, loader, xCharList, yCharList
from mask import maskPad, maskTril
from model import Transformer

def train():
    model = Transformer()
    lossFun = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.StepLR(optim, 3, gamma=0.5)
    for epoch in range(1):
        for i, (x, y) in enumerate(loader):
            # x = [8, 50]
            # y = [8, 51]
            # 在训练时,是拿y的每一个字符输入,预测下一个字符,所以不需要最后一个字
            # [8, 50, 39]
            pred = model(x, y[:, :-1])
            # [8, 50, 39] -> [400, 39]
            pred = pred.reshape(-1, 39)
            # [8, 51] -> [400]
            y = y[:, 1:].reshape(-1)
            # 忽略pad的部分
            select = y != yDict["<PAD>"]
            pred = pred[select]
            y = y[select]
            loss = lossFun(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if i % 100 == 0:
                # [select, 39] -> [select]
                pred = pred.argmax(1)
                correct = (pred == y).sum().item()
                accuracy = correct / len(pred)
                lr = optim.param_groups[0]['lr']
                print("epoch:{}, step:{}, lr:{:.4f}, loss:{:.4f}, accuracy:{:.4f}, ".format(epoch, i, lr, loss.item(), accuracy))
        sched.step()

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
    }, './transformer.pt')


def predict(model: Transformer, x):
    model.eval()
    # x = [1, 50]
    maskPadX = maskPad(x)
    # 初始化输出y    [1, 50]
    target = [yDict['<SOS>']] + [yDict['<PAD>']] * 49
    target = torch.LongTensor(target).unsqueeze(0)
    # embedding x   [1, 50] -> [1, 50, 32]
    x = model.embedX(x)
    # 编码器层, 维度不变  [1, 50, 32] -> [1, 50, 32]
    x = model.encoder(x, maskPadX)
    # 逐个生成第1至49个词
    for i in range(49):
        y = target  # [1, 50]
        # 掩码 [1, 1, 50, 50]
        maskTrilY = maskTril(y)
        # embedding y   [1, 50] -> [1, 50, 32]
        y = model.embedY(y)
        # 解码器层 [1, 50, 32],[1, 50, 32] -> [1, 50, 32]
        y = model.decoder(x, y, maskPadX, maskTrilY)
        # 全链接，对每个字符输出39类的分类, [1, 50, 32] -> [1, 50, 39]
        out = model.fcOut(y)
        # 取出当前词的输出  [1, 50, 39] -> [1, 39]
        out = out[:, i, :]
        # 取出分类结果
        out = out.argmax(dim=1).detach()
        target[:,i + 1] = out
    return target

def predictTest():
    model = Transformer()
    checkpoint = torch.load('./transformer.pt', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    for i, (x, y) in enumerate(loader):
        break

    for i in range(x.shape[0]):
        print("{} inputX {}".format(i, ''.join([xCharList[i] for i in x[i].tolist()])))
        print("{} outputY {}".format(i, ''.join([yCharList[i] for i in y[i].tolist()])))
        predY = predict(model, x[i].unsqueeze(0))[0]
        print("{}  predY  {}".format(i, ''.join([yCharList[i] for i in predY.tolist()])))
        print('----------------------------------')
def main():
    # train()
    predictTest()


if __name__ == "__main__":
    main()
