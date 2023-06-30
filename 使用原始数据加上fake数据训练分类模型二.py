import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from utils.data_reader import LoadData  # 数据读取
from utils.network_structure import classification_model  # 网路结构
from utils.functional_functions import init_weights  # 参数初始化

# 数据目录
route = 'data\MNIST'  # 数据目录
fake_route = 'data\MNIST_fake'  # 生成的fake样本的保存目录
result_save_path = 'model/CNN_model2'  # 模型和loss图的保存目录
drop_last = False  # 不够一个批次的数据是否舍弃掉，数据量多可以选择True
if not os.path.exists(result_save_path):
    os.mkdir(result_save_path)  # 如果没有保存路径的目录文件夹则进行创建

# 训练相关的参数
lr = 0.002  # 判别器学习率
batch_size = 100  # 一个批次的大小
num_epoch = 100  # 训练迭代次数
output_loss_Interval_ratio = 1  # 间隔多少个epoch打印一次损失
test_interval = 10  # 间隔多少个epoch测试一次准确率

# 网络结构相关的参数
nc = 1  # 输入通道数，RGB为3，GRAY为1

criterion = nn.CrossEntropyLoss()
model = classification_model(n_classes=10, number_of_channels=nc).cuda()
optimer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# 初始化训练数据读取器
train_dataset = ConcatDataset([LoadData(os.path.join(route, 'train', str(
    number)), number_of_channels=nc) for number in range(0, 10)] +
    [LoadData(os.path.join(fake_route, 'train', str(
        number)), number_of_channels=nc) for number in range(0, 10)])  # dataset
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                          shuffle=True, drop_last=drop_last)  # dataloader

val_dataset = ConcatDataset([LoadData(os.path.join(route, 'val', str(
    number)), number_of_channels=nc) for number in range(0, 10)])  # dataset
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                        shuffle=True, drop_last=drop_last)  # dataloader

# 初始化模型
model.apply(init_weights)

loss_list = []  # 保存每一个epoch的损失值
acc_list = []  # 保存每一个epoch的准确率
for epoch in range(0, num_epoch+1):  # 迭代num_epoch个epoch
    # 训练
    model.train()
    batch_loss = 0  # 累加每个epoch中全部batch的损失值，最后平均得到每个epoch的损失值
    # 每个batch_size的图片
    for img, label in tqdm(train_loader, desc=f'Epoch[{epoch}] train'):
        label = torch.as_tensor(label, dtype=torch.long).cuda()
        output = model(img.cuda())  # 前向传播
        loss = criterion(output, label)  # 计算loss
        optimer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimer.step()  # 参数更新
        batch_loss += loss  # 累加loss

    # 保存损失值为列表,将所有batch累加的损失值除以batch数即该轮epoch的损失值
    loss_list.append(batch_loss.item()/len(train_loader))

    # 测试
    if epoch % test_interval == 0:  # 间隔test_interval个epoch测试一次准确率
        model.eval()
        batch_acc = 0
        # 每个batch_size的图片
        for img, label in tqdm(val_loader, desc=f'Epoch[{epoch}] test'):
            label = torch.as_tensor(label, dtype=torch.long).cuda()
            prediction_output = model(img.cuda())
            batch_acc += sum(torch.argmax(prediction_output,
                             dim=1) == label)/len(img)

        # 将该轮的测试准确率保存到列表当中
        acc_list.append(batch_acc.item()/len(val_loader))

    # 打印训练的损失和测试的准确率  #间隔output_loss_Interval_ratio个epoch打印一次损失
    if epoch % output_loss_Interval_ratio == 0:
        print('Epoch[{}/{}],loss:{:.6f}'.format(
            epoch, num_epoch,
            batch_loss.item()/len(train_loader)
        ))  # 打印每个epoch的损失值

    # 如果做了测试，则打印准确率
    if epoch % test_interval == 0:
        print('Epoch[{}/{}],acc:{:.6f}'.format(
            epoch, num_epoch,
            acc_list[-1]
        ))  # 打印每个epoch的损失值

    # 保存loss图像
    plt.plot(range(len(loss_list)), loss_list, label="loss")
    plt.plot([i*test_interval for i in range(len(acc_list))],
             acc_list, label="acc")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join(result_save_path, 'loss.jpg'))
    plt.clf()

    # 创建保存模型和loss图的目录
    if not os.path.exists(os.path.join(result_save_path)):
        os.mkdir(os.path.join(result_save_path))

    # 保存模型
    torch.save(model, os.path.join(result_save_path, 'last.pth'))
