import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_class):  # num_class是分类数
        super(CNN, self).__init__()
        # 特征提取
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 定义全连接层， 做分类
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, num_class)
        )

    # 前向传播
    def forward(self, x):
        x = self.features(x)  # 先将特征图进行特征提取
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
