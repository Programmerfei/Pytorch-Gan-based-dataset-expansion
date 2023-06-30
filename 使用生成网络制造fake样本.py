import os
from tqdm import tqdm

import torch
from torchvision.utils import save_image 

img_number=500  #每一个数字生成多少张fake图片

result_save_path='model/GAN_model'  #训练好的生成网络模型的目录
fakedata_save_path='data/MNIST_fake/train/'   #生成的fake图片保存目录

if not os.path.exists(fakedata_save_path):
    os.makedirs(fakedata_save_path)

for number in range(0,10):
    g=torch.load(os.path.join(result_save_path,str(number),str(number)+'_g_last.pth'))  #加载模型
    fake_save_dir=os.path.join(fakedata_save_path,str(number))    #保存图片的目录路径
    if not os.path.exists(fake_save_dir):  #如果没有这个路径则创建
        os.mkdir(fake_save_dir)
    
    g.eval()#进入验证模式，不用计算梯度和参数更新
    g_input=next(g.children())[0].in_channels  #获取模型的输入通道数

    for i in tqdm(range(img_number),desc=f'number{number}'):
        z = torch.randn(1,g_input,1,1).cuda()  # 随机生成一些噪声
        fake_img = g(z).detach()  # 随机噪声放入生成网络中，生成一张假的图片。 # 避免梯度传到G，因为G不用更新, detach分离
        save_image(fake_img,os.path.join(fake_save_dir,
                        str(number)+'_fake_'+str(i)+'.jpg'))  #保存fake样本
        
    