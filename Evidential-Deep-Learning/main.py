import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets.mnist import MNIST  # 从torchvision库中导入MNIST数据集类
import torchvision.transforms as transforms  # 导入图像转换模块
from torch.utils.data import DataLoader  # 导入数据加载器类
from torch.autograd import Variable

import numpy as np
import argparse  # 导入命令行参数解析模块
from matplotlib import pyplot as plt
from PIL import Image

from helpers import get_device, rotate_img, one_hot_embedding  # 导入辅助函数
from data import dataloaders, digit_one  # 导入数据加载相关函数
from train import train_model  # 导入训练模型相关函数
from test import rotating_image_classification, test_single_image  # 导入测试模型相关函数
from losses import edl_mse_loss, edl_digamma_loss, edl_log_loss, relu_evidence  # 导入损失函数相关函数
from lenet import LeNet  # 导入自定义的LeNet模型类


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--train", action="store_true", help="To train the network."
    )
    mode_group.add_argument("--test", action="store_true", help="To test the network.")
    mode_group.add_argument(
        "--examples", action="store_true", help="To example MNIST data."
    )
    parser.add_argument(
        "--epochs", default=10, type=int, help="Desired number of epochs."
    )
    parser.add_argument(
        "--dropout", action="store_true", help="Whether to use dropout or not."
    )
    parser.add_argument(
        "--uncertainty", action="store_true", help="Use uncertainty or not."
    )
    uncertainty_type_group = parser.add_mutually_exclusive_group()
    uncertainty_type_group.add_argument(
        "--mse",
        action="store_true",
        help="Set this argument when using uncertainty. Sets loss function to Expected Mean Square Error.",
    )
    uncertainty_type_group.add_argument(
        "--digamma",
        action="store_true",
        help="Set this argument when using uncertainty. Sets loss function to Expected Cross Entropy.",
    )
    uncertainty_type_group.add_argument(
        "--log",
        action="store_true",
        help="Set this argument when using uncertainty. Sets loss function to Negative Log of the Expected Likelihood.",
    )
    args = parser.parse_args()

    if args.examples:
        examples = enumerate(dataloaders["val"])  # 获取验证集数据迭代器
        batch_idx, (example_data, example_targets) = next(examples)
        fig = plt.figure()
        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.tight_layout()
            plt.imshow(example_data[i][0], cmap="gray", interpolation="none")
            plt.title("Ground Truth: {}".format(example_targets[i]))
            plt.xticks([])
            plt.yticks([])
        plt.savefig("./images/examples.jpg")

    elif args.train:
        num_epochs = args.epochs
        use_uncertainty = args.uncertainty
        num_classes = 10

        model = LeNet(dropout=args.dropout)  # 创建LeNet模型实例

        if use_uncertainty:
            if args.digamma:
                criterion = edl_digamma_loss  # 使用对数Gamma函数作为损失函数
            elif args.log:
                criterion = edl_log_loss  # 使用对数似然损失函数
            elif args.mse:
                criterion = edl_mse_loss  # 使用均方误差损失函数
            else:
                parser.error("--uncertainty requires --mse, --log or --digamma.")
        else:
            criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数

        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.005)

        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        device = get_device()
        model = model.to(device)  # 将模型移至指定设备

        model, metrics = train_model(
            model,
            dataloaders,
            num_classes,
            criterion,
            optimizer,
            scheduler=exp_lr_scheduler,
            num_epochs=num_epochs,
            device=device,
            uncertainty=use_uncertainty,
        )  # 训练模型

        state = {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        if use_uncertainty:
            if args.digamma:
                torch.save(state, "./results/model_uncertainty_digamma.pt")
                print("Saved: ./results/model_uncertainty_digamma.pt")
            if args.log:
                torch.save(state, "./results/model_uncertainty_log.pt")
                print("Saved: ./results/model_uncertainty_log.pt")
            if args.mse:
                torch.save(state, "./results/model_uncertainty_mse.pt")
                print("Saved: ./results/model_uncertainty_mse.pt")
        else:
            torch.save(state, "./results/model.pt")
            print("Saved: ./results/model.pt")

    elif args.test:

        use_uncertainty = args.uncertainty
        device = get_device()
        model = LeNet()
        model = model.to(device)
        optimizer = optim.Adam(model.parameters())

        if use_uncertainty:
            if args.digamma:
                checkpoint = torch.load("./results/model_uncertainty_digamma.pt")
                filename = "./results/rotate_uncertainty_digamma.jpg"
            if args.log:
                checkpoint = torch.load("./results/model_uncertainty_log.pt")
                filename = "./results/rotate_uncertainty_log.jpg"
            if args.mse:
                checkpoint = torch.load("./results/model_uncertainty_mse.pt")
                filename = "./results/rotate_uncertainty_mse.jpg"

        else:
            checkpoint = torch.load("./results/model.pt")
            filename = "./results/rotate.jpg"

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        model.eval()

        rotating_image_classification(
            model, digit_one, filename, uncertainty=use_uncertainty
        )  # 旋转图像分类和不确定性可视化

        test_single_image(model, "./data/one.jpg", uncertainty=use_uncertainty)  # 测试单张图像
        test_single_image(model, "./data/yoda.jpg", uncertainty=use_uncertainty)  # 测试另一张图像


if __name__ == "__main__":
    main()
