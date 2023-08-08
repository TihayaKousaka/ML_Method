import os
from pathlib import Path
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import cv2
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import img_to_tensor
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Normalize,
    Compose,
    PadIfNeeded,
    RandomCrop,
    CenterCrop,
    Resize
)
from torchvision.transforms import ToTensor
from torchvision import transforms

target_img_size = 256

# 图像转换函数，用于数据增强
def image_transform(p=1):
    return Compose([
        Resize(target_img_size, target_img_size, cv2.INTER_LINEAR),
        # Normalize(p=1)
    ], p=p)

# 掩码转换函数，用于数据增强
def mask_transform(p=1):
    return Compose([
        Resize(target_img_size, target_img_size, cv2.INTER_NEAREST)
    ], p=p)

# 从路径中提取文件名的函数
def get_id(input_img_path):
    data_path = Path(input_img_path) # 将输入路径转换为Path对象
    pred_file_name = []
    pred_file_name.append(data_path)
    return pred_file_name

# 获取数据集的拆分（训练集、验证集和测试集）
def get_split():
    # TODO：不要硬编码绝对路径
    data_path = Path('C:/Users/tom/Desktop/summer_research/ESD_seg') # 数据集的根路径

    seed = 0
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    train_data_file_names = []
    val_data_file_names = []
    test_data_file_names = []

    # 指定数据集拆分
    train_ids =  ['06', '24', '15', '27', '09', '01', '12', '31', '29', '28', '33', '35', '08', '05', '32', '11', '16']
    val_ids = ['30', '14', '36', '25']
    test_ids =  ['26', '13', '03', '23', '34', '18']

    for data_id in train_ids:
        train_data_file_names += list((data_path / (str(data_id)) / 'image').glob('*'))
    for data_id in val_ids:
        val_data_file_names += list((data_path / (str(data_id)) / 'image').glob('*'))
    for data_id in test_ids:
        test_data_file_names += list((data_path / (str(data_id)) / 'image').glob('*'))

    return train_data_file_names, val_data_file_names, test_data_file_names

# 自定义数据集类
class ESD_Dataset(Dataset):
    def __init__(self, file_names, ids=False):
        self.file_names = file_names
        self.image_transform = image_transform()
        self.mask_transform = mask_transform()
        self.ids = ids
        self.transforms = ToTensor()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]

        image = load_image(img_file_name)  # 加载图像
        mask = load_mask(img_file_name)  # 加载掩码

        data = {"image": image, "mask": mask}
        augmented = self.mask_transform(**data)  # 对掩码进行数据增强
        mask = augmented["mask"]
        image = self.image_transform(image=image)

        image = image['image']
        image = self.transforms(image)
        label = torch.from_numpy(mask).long()
        sample = {'image': image, 'label': label, 'id': str(img_file_name).split('\\')[-1]}
        if self.ids: 
            return sample['image'], sample['label'], sample['id']
        return sample['image'], sample['label']

# 加载图像
def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 加载掩码
def load_mask(path):
    mask_folder = 'mask'
    path = str(path).replace('image', mask_folder)
    identifier = path.split('/')[-1]
    path = path.replace(identifier, identifier[:-4] + '_mask' + '.png')
    mask = cv2.imread(str(path), 0)
    mask[mask == 255] = 4
    mask[mask == 212] = 0
    mask[mask == 170] = 0
    mask[mask == 128] = 3
    mask[mask == 85] = 2
    mask[mask == 42] = 1
    
    return mask.astype(np.uint8)
