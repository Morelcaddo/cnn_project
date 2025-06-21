import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_class):  # num_class是分类数
        super(CNN, self).__init__()
        # 特征提取
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8
        )

        # 计算全连接层的输入大小
        # 输入图像大小为 64x64，经过三次池化后变为 8x8
        # 因此，全连接层的输入大小为 64 * 8 * 8
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_class)
        )

    # 前向传播
    def forward(self, x):
        x = self.features(x)  # 先将特征图进行特征提取
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
