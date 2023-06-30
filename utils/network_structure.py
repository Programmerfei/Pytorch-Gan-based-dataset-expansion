import torch.nn as nn


ndf=64 #判别网络卷积核个数的倍数
ngf=64 #生成网络卷积核个数的倍数


"""
关于转置卷积：
当padding=0时,卷积核刚好和输入边缘相交一个单位。因此pandding可以理解为卷积核向中心移动的步数。 
同时stride也不再是kernel移动的步数,变为输入单元彼此散开的步数,当stride等于1时,中间没有间隔。
"""

#生成器网络G
class generator(nn.Module):
    def __init__(self,noise_number,number_of_channels):
        """
        noise_number:输入噪声点个数
        number_of_channels:生成图像通道数
        """
        super(generator,self).__init__()
        self.gen = nn.Sequential(
            # 输入大小  batch x noise_number x 1 * 1
            nn.ConvTranspose2d(noise_number , ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 输入大小 batch x (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 输入大小 batch x (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 2, ngf , 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf ),
            nn.ReLU(True),
            # 输入大小 batch x (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf   , number_of_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # 输出大小 batch x (nc) x 64 x 64
       )

    def forward(self, x):
        out = self.gen(x)
        return out
    
#判别器网络D
class discriminator(nn.Module):
    def __init__(self,number_of_channels):
        """
        number_of_channels:输入图像通道数
        """
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            # 输入大小 batch x g_d_nc x 64*64
            nn.Conv2d(number_of_channels, ndf  , 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf ),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入大小 batch x ndf x 32*32
            nn.Conv2d(ndf , ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入大小 batch x (ndf*2) x 16*16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入大小 batch x (ndf*8) x 4*4
            nn.Conv2d(ndf * 4 , 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # 输出大小 batch x 1 x 1*1
        )

    def forward(self, x):
        x=self.dis(x).view(x.shape[0],-1)
        return x
    
#分类网络CNN
class classification_model(nn.Module):
    def __init__(self,n_classes,number_of_channels):
        """
        n_classes:类别数
        """
        super(classification_model,self).__init__()
        self.structure=nn.Sequential(
            nn.Conv2d(number_of_channels, 6, kernel_size=5, stride=1, padding=2),  # (m,6,28,28)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # (m,6,14,14)

            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),  # (m,16,10,10)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # (m,16,5,5)

            nn.Conv2d(16, n_classes, kernel_size=5, stride=1, padding=0),  # (m,10,1,1)
            nn.Softmax(dim=1)
        )   
    
    def forward(self,x):
        out=self.structure(x)
        out=out.reshape(out.shape[0],-1)
        return out