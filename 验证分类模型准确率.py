import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, ConcatDataset

from utils.data_reader import LoadData  # 数据读取

route = 'data/MNIST'  # 数据目录
model_path='model/CNN_model1'  #模型路径
batch_size = 100  # 一个批次的大小
drop_last = False  # 不够一个批次的数据是否舍弃掉，数据量多可以选择True
nc = 1  # 输入通道数，RGB为3，GRAY为1

val_dataset = ConcatDataset([LoadData(os.path.join(route, 'val', str(
    number)), number_of_channels=nc) for number in range(0, 10)])  # dataset
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                        shuffle=True, drop_last=drop_last)  # dataloader

model=torch.load(os.path.join(model_path,'last.pth'))

batch_acc=0
for img,label in tqdm(val_loader):
    label = torch.as_tensor(label, dtype=torch.long).cuda()
    prediction_output = model(img.cuda())
    batch_acc += sum(torch.argmax(prediction_output,
                        dim=1) == label)/len(img)
print(f"acc:{batch_acc/len(val_loader)}")