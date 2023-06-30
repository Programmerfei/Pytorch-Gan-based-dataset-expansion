import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

MNIST_data_dir = 'data/MNIST/raw/'  # MNIST数据文件路径，这里存放的是二进制文件
train_val_data_dir = 'data/MNIST/'  # train和val的数据保存路径，train是6w张数据，val是1w张数据
Number_of_requirements = 500  # 每个数字取多少张数据作为训练数据及测试数据，解析到足量则提前结束


def read_idx(filename):
    """
    二进制文件解析函数
    filename:二进制文件路径
    """
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def save_img(data, labels, t_v):
    """
    图片保存函数
    data: 二进制文件解析出来的图片数据
    labels: 标签
    t_v: train或val
    """
    count_dict = {}
    for i in tqdm(range(len(data)), desc=t_v):
        label = labels[i]
        folder = os.path.join(t_v, str(label))
        if not os.path.exists(folder):
            os.makedirs(folder)
        if sum(count_dict.values()) == 10*Number_of_requirements:  # 如果每个数字都达到需求个数，则结束
            break
        # 如果这个数字的个数达到要求则跳过这个数字的保存
        if str(label) in count_dict and count_dict[str(label)] == Number_of_requirements:
            continue
        # if os.path.exists(os.path.join(folder, f'image_{i}.png')):   #如果图片存在先删除之前保存的，再重新保存新的图片（防止之前保存的有问题）
        #     os.remove(os.path.join(folder, f'image_{i}.png'))
        cv2.imwrite(os.path.join(folder, f'image_{i}.jpg'), data[i])
        # 保存一次图片，这个数字的计数+1，如果字典中没有，即为该数字的第一张图，赋值为1
        count_dict[str(label)] = count_dict[str(label)] + \
            1 if str(label) in count_dict else 1
    print('数量已达要求,停止解析:\n', count_dict)


if __name__ == '__main__':
    for data_path, label_path, t_v in zip(['train-images-idx3-ubyte', 't10k-images-idx3-ubyte'],
                                          ['train-labels-idx1-ubyte',
                                              't10k-labels-idx1-ubyte'],
                                          ['train', 'val']):
        data = read_idx(os.path.join(MNIST_data_dir, data_path))  # 解析图片文件
        labels = read_idx(os.path.join(
            MNIST_data_dir, label_path))  # 解析label文件
        save_img(data, labels, os.path.join(train_val_data_dir, t_v))  # 保存图片
