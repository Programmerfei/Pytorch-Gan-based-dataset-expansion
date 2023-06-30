# -*- coding=utf-8 -*-
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from utils.data_reader import LoadData  # 数据读取
from utils.network_structure import generator, discriminator  # 网路结构
from utils.functional_functions import init_weights  # 参数初始化

# 数据目录
route = 'data\MNIST'  # 数据目录
result_save_path = 'model/GAN_model'  # 模型和训练过程中生成的fake样本的保存目录
drop_last = False  # 不够一个批次的数据是否舍弃掉，数据量多可以选择True
if not os.path.exists(result_save_path):
    os.mkdir(result_save_path)  # 如果没有保存路径的目录文件夹则进行创建

# 训练相关的参数
lr_d = 0.002  # 判别器学习率
lr_g = 0.002  # 生成器学习率
batch_size = 100  # 一个批次的大小
num_epoch = 300  # 训练迭代次数
output_loss_Interval_ratio = 10  # 间隔多少个epoch打印一次损失
save_model_Interval_ratio = 100  # 间隔多少个epoch保存一次训练过程中的fake图片

# 网络结构相关的参数
g_d_nc = 1  # d的输入通道和g的输出通道，RGB为3，GRAY为1
g_input = 100  # g的输入噪声点个数

# 定义loss的度量方式
criterion = nn.BCELoss()  # 单目标二分类交叉熵函数
# 实例化生成器和判别器
d = discriminator(number_of_channels=g_d_nc).cuda()
g = generator(noise_number=g_input,
              number_of_channels=g_d_nc).cuda()  # 模型迁移至GPU
# 定义 优化函数 学习率
d_optimizer = torch.optim.Adam(
    d.parameters(), lr=lr_d, betas=(0.5, 0.999))  # Adam优化器
g_optimizer = torch.optim.Adam(g.parameters(), lr=lr_g, betas=(0.5, 0.999))

# 调试代码，用于验证输入图像大小和g网络结构的适配性
# # 下面注释的这几行代码用于调试g网络层输入输出大小用
# z = torch.randn(batch_size,g_input,1,1).cuda()  # 随机生成一些噪声
# for i in g.gen:
#     print(i(z).shape)
#     z=i(z)

# # 下面注释的这几行代码用于调试d网络层输入输出大小用
# z = torch.randn(batch_size,g_input,1,1).cuda()  # 随机生成一些噪声
# fake_img=g(z)
# for i in d.dis:
#     print(i(fake_img).shape)
#     fake_img=i(fake_img)

for number in range(0, 10):  # 0-9每一个数字单独训练
    # 初始化网络每一层的参数
    d.apply(init_weights), g.apply(init_weights)

    # #恢复训练
    # g=torch.load(os.path.join(result_save_path,str(number),str(number)+'_g__last.pth'))
    # d=torch.load(os.path.join(result_save_path,str(number),str(number)+'_d__last.pth'))

    # 初始化训练数据读取器
    train_dataset = LoadData(os.path.join(route, 'train', str(
        number)), number_of_channels=g_d_nc)  # dataset
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=drop_last)  # dataloader

    loss_list_g, loss_list_d = [], []  # 保存每一个epoch的损失值
    for epoch in tqdm(range(0, num_epoch+1), desc='epoch'):  # 迭代num_epoch个epoch
        batch_d_loss, batch_g_loss = 0, 0  # 累加每个epoch中全部batch的损失值，最后平均得到每个epoch的损失值
        for img, label in train_loader:  # 每个batch_size的图片
            img_number = len(img)  # 该批次有多少张图片
            real_img = img.cuda()  # 将tensor放入cuda中
            real_label = torch.ones(img_number).cuda()  # 定义真实的图片label为1
            fake_label = torch.zeros(img_number).cuda()  # 定义假的图片的label为0

            # ==================训练判别器==================
            # 分为两部分：1、真的图像判别为真；2、假的图像判别为假
            # 计算真实图片的损失
            real_out = d(real_img)  # 将真实图片放入判别器中
            real_label = real_label.reshape([-1, 1])  # shape (n) -> (n,1)
            d_loss_real = criterion(real_out, real_label)  # 得到真实图片的loss
            real_scores = real_out  # 得到真实图片的判别值，输出的值越接近1越好
            # 计算假的图片的损失
            z = torch.randn(img_number, g_input, 1, 1).cuda()  # 随机生成一些噪声
            # 随机噪声放入生成网络中，生成一张假的图片。 # 避免梯度传到G，因为G不用更新, detach分离
            fake_img = g(z).detach()
            fake_out = d(fake_img)  # 判别器判断假的图片，
            fake_label = fake_label.reshape([-1, 1])  # shape (n) -> (n,1)
            d_loss_fake = criterion(fake_out, fake_label)  # 得到假的图片的loss
            fake_scores = fake_out  # 得到假图片的判别值，对于判别器来说，假图片的损失越接近0越好
            # 合计判别器的总损失
            d_loss = d_loss_real + d_loss_fake  # 损失包括判真损失和判假损失
            # 反向传播，参数更新
            d_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
            d_loss.backward()  # 将误差反向传播
            d_optimizer.step()  # 更新参数

            # ==================训练生成器==================
            # 原理：目的是希望生成的假的图片被判别器判断为真的图片，
            # 在此过程中，将判别器固定，将假的图片传入判别器的结果与真实的label对应，
            # 反向传播更新的参数是生成网络里面的参数，
            # 这样可以通过更新生成网络里面的参数，来训练网络，使得生成的图片让判别器以为是真的
            # 这样就达到了对抗的目的
            # 计算假的图片的损失
            z = torch.randn(img_number, g_input, 1, 1).cuda()  # 得到随机噪声
            fake_img = g(z)  # 随机噪声输入到生成器中，得到一副假的图片
            output = d(fake_img)  # 经过判别器得到的结果
            g_loss = criterion(output, real_label)  # 得到的假的图片与真实的图片的label的loss
            # 反向传播，参数更新
            g_optimizer.zero_grad()  # 梯度归0
            g_loss.backward()  # 进行反向传播
            g_optimizer.step()  # .step()一般用在反向传播后面,用于更新生成网络的参数

            # ==================累加总损失值，后面进行损失值可视化==================
            batch_d_loss += d_loss  # 累加每一个batch的损失值
            batch_g_loss += g_loss  # 累加每一个batch的损失值

        # # 调整学习率,当判别器损失足够小的时候，大幅度降低d的学习率，防止d过于完美，导致g无法训练(增加epoch次数可以开启)
        # if d_loss < 0.5:
        #     for i in d_optimizer.param_groups:
        #         i['lr']=lr_d/10

        # 将该轮的损失函数值保存到列表当中
        # 保存g损失值为列表,将所有batch累加的损失值除以batch数即该轮epoch的损失值
        loss_list_g.append(batch_g_loss.item()/len(train_loader))
        loss_list_d.append(batch_d_loss.item()/len(train_loader))  # 保存d损失值为列表

        # 打印中间的损失  #间隔output_loss_Interval_ratio个epoch打印一次损失
        if epoch % output_loss_Interval_ratio == 0:
            print('\nnumber:{} Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '
                  'D real: {:.6f},D fake: {:.6f}'.format(
                      number, epoch, num_epoch,
                      batch_d_loss.item()/len(train_loader),
                      batch_g_loss.item()/len(train_loader),
                      real_scores.data.mean(),
                      fake_scores.data.mean()
                  ))  # 打印每个epoch的d和g损失值（越小越好）和d的判别值（real越接近1越好，fake越接近0越好）

        # 创建保存模型和生成fake样本以及loss图的目录
        if not os.path.exists(os.path.join(result_save_path, str(number))):
            os.mkdir(os.path.join(result_save_path, str(number)))

        # 保存生成的fake图片，间隔save_model_Interval_ratio个epoch保存一次
        if epoch % save_model_Interval_ratio == 0:
            save_image(fake_img, os.path.join(result_save_path, str(number),
                                              str(number)+'_fake_epoch'+str(epoch)+'.jpg'))

        # 保存模型,for分别保存g和d，每个epoch都保存一次last.pth
        for g_or_d, g_d_name in zip([g, d], ['_g_', '_d_']):
            torch.save(g_or_d, os.path.join(result_save_path,
                       str(number), str(number)+g_d_name+'last.pth'))

        # 保存loss图像
        plt.plot(range(len(loss_list_g)), loss_list_g, label="g_loss")
        plt.plot(range(len(loss_list_d)), loss_list_d, label="d_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(os.path.join(result_save_path, str(number), 'loss.jpg'))
        plt.clf()

    print('\n')
