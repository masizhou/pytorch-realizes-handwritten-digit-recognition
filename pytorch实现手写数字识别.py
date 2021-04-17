import os
import torch
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


BATCH_SIZE = 128



# 1、准备数据集
def get_dataloader(train=True):
    # 准备数据集，其中0.1307, 0.3081为MNIST数据集的均值和标准差，这样操作能够对其进行标准化
    # 因为MNIST只有一个通道（黑白色），所以列表中只有一个
    transforms_fn = transform=Compose([ToTensor(),  # 先转化为Tensor
                                       Normalize(mean=[0.1307], std=[0.3081])]) # 再进行正则化

    dataset = MNIST("./data", train=train, download=False, transform=transforms_fn)

    # 准备数据迭代器
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    return data_loader

# 2、构建模型
class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.fc1 = nn.Linear(1*28*28, 28)
        self.fc2 = nn.Linear(28, 10)

    def forward(self, input):
        """
        :param x:[batch_size, 1, 28, 28]
        :return:
        """
        # 1、修改形状
        x = input.view(input.size(0), 1*28*28)
        # input = input.view(-1, 1*28*28)

        # 2、进行全连接操作
        x = self.fc1(x)

        # 3、进行激活函数的处理,形状不会变
        F.relu(x)

        # 4、输出层
        out = self.fc2(x)

        return F.log_softmax(out, dim=-1)

model = MnistModel()
optimizer = Adam(model.parameters(), lr=0.001)

if os.path.exists("./model/model.pkl"):
    # 加载模型参数
    model.load_state_dict(torch.load("./model/model.pkl"))
    # 加载优化器的参数
    optimizer.load_state_dict(torch.load("./model/optimizer.pkl"))

def train(epoch):
    """实现训练过程"""
    mode = True
    model.train(mode=mode ) # 模型设置为训练模式

    data_loader = get_dataloader()
    for idx, (input, target) in enumerate(data_loader): # 每一轮里面的数据进行遍历
        optimizer.zero_grad() # 梯度清零
        output = model(input) # 调用模型，得到预测值
        loss = F.nll_loss(output, target) # 得到损失
        loss.backward() # 反向传播
        optimizer.step() # 梯度更新
        if idx % 10 == 0:
            print("epoch:",epoch," idx:",idx, " loss:",loss.item())

        # 模型的保存
        if idx % 100 == 0:
            torch.save(model.state_dict(), "./model/model.pkl")
            torch.save(optimizer.state_dict(), "./model/optimizer.pkl")

def test():
    loss_list = []
    acc_list = []

    test_dataloader = get_dataloader(train=False)
    for inx, (input, target) in enumerate(test_dataloader):
        with torch.no_grad():
            output = model(input)
            cur_loss = F.nll_loss(output, target)
            loss_list.append(cur_loss)
            # print(type(output))
            # 计算准确率
            # output [batch_size, 10], target [batch_size]
            pred = output.max(dim=-1)[-1] # tensor的max/min返回两个值：第一个是值，第二个是对应的索引,因此加[-1]表示取索引,和argmax一样
            # print(pred)
            cur_acc = pred.eq(target).float().mean()
            acc_list.append(cur_acc)
    print("平均准确率:", np.mean(acc_list), "平均损失:", np.mean(loss_list))


if __name__ == "__main__":
    # for i in range(3): # 训练三轮
    #     train(i)
    test()










