import torch
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 定义训练集数据集，指定保存路径、下载标志、训练集标志和数据预处理操作
data_train = MNIST("./data/mnist",
                   download=True,
                   train=True,
                   transform=transforms.Compose([transforms.ToTensor()]))

# 定义验证集数据集，指定保存路径、训练集标志、下载标志和数据预处理操作
data_val = MNIST("./data/mnist",
                 train=False,
                 download=True,
                 transform=transforms.Compose([transforms.ToTensor()]))

# 创建训练集数据加载器，指定数据集、批大小、随机打乱标志和并行加载的工作线程数
dataloader_train = DataLoader(
    data_train, batch_size=1000, shuffle=True, num_workers=8)

# 创建验证集数据加载器，指定数据集、批大小和并行加载的工作线程数
dataloader_val = DataLoader(data_val, batch_size=1000, num_workers=8)

# 将训练集和验证集数据加载器保存到一个字典中
dataloaders = {
    "train": dataloader_train,
    "val": dataloader_val,
}

# 从验证集数据中获取第6个样本（下标从0开始），其中的第一个元素为图像数据，第二个元素为标签（此处用下划线表示不使用）
digit_one, _ = data_val[5]
