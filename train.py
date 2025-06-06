import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets  # 数据集和数据变换
from tqdm import tqdm  # 训练进度条

from model.cnn import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载测试集和训练集
train_set = datasets.ImageFolder(root=os.path.join(r"data", "train"),
                                 transform=train_transform)

test_set = datasets.ImageFolder(root=os.path.join(r"data", 'test'),
                                transform=test_transform)

# train_set传入的训练集，batch批次训练的图片数量，num_workers数据加载多线程， shuffle为True代表打乱加载数据
train_loader = DataLoader(train_set, batch_size=32, num_workers=0, shuffle=True)

test_loader = DataLoader(train_set, batch_size=32, num_workers=0, shuffle=True)


def train(model, trainLoader, criterion, optimizer, num_epochs):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(trainLoader, desc=f"epoch:{epoch + 1}/{num_epochs}", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据传到设别设备上
            optimizer.zero_grad()  # 梯度清零
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # loss的计算
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            running_loss += loss.item() * inputs.size(0)  # 用loss乘批次大小，得到该批次的loss
        epoch_loss = running_loss / len(trainLoader.dataset)  # 总损失除数据集大小，为我们每轮的损失
        print(f"Epoch {epoch + 1}/{num_epochs},Training_Loss: {epoch_loss:.4f}")

        accuracy = evaluate(model, test_loader, criterion)
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), save_path)
            print("model saved with best acc", best_acc)


def evaluate(model, testLoader, criterion):
    model.eval()  # 指定模型为验证模式
    test_loss = 0.0  # 初始的测试数量为0
    correct = 0  # 正确样本数量为0
    total = 0  # 总样本数量
    with torch.no_grad():  # 在评估模式下不需要计算梯度
        for inputs, labels in testLoader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # 计算损失
            test_loss = test_loss + loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)  # 获取模型预测的最大值
            total += labels.size(0)  # 计算样本总数量
            correct = correct + (predicted == labels).sum().item()  # 正确样本数累加

    avg_loss = test_loss / len(testLoader.dataset)  # 计算平均loss
    accuracy = 100.0 * correct / total  # 计算准确率
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    return accuracy


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    num_epochs = 10
    learning_rate = 0.001
    num_classes = 62
    save_path = r"model_pth\best.pth"
    model = CNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train(model, train_loader, criterion, optimizer, num_epochs)
    evaluate(model, test_loader, criterion)
