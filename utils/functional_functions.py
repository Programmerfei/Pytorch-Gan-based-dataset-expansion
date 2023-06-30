import torch.nn as nn

#初始化网络参数函数，用于下一个数字开始训练之前
def init_weights(m):
    if hasattr(m,'weight'):
        nn.init.uniform_(m.weight,-0.1,0.1)