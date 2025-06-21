import torch
from PIL import Image
from torchvision import transforms

from model.cnn import CNN


def identification(path):
    model = CNN(num_class=26)
    model.load_state_dict(torch.load('model_pth/best4.0_20.pth'))
    model.eval()

    # 定义数据预处理变换，必须与训练时的预处理一致
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 加载一张新图像
    # image = Image.open('data/letter_1.jpg')  # 替换为你的图像路径
    image = Image.open(path)

    # 对图像进行预处理
    input_tensor = transform(image).unsqueeze(0)  # 添加批次维度，因为模型期望输入是 [batch, channels, height, width]

    # 预测
    with torch.no_grad():  # 不需要计算梯度
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)  # 获取预测的类别索引

    indexes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
               'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    print(f"Predicted class index: {predicted_class.item()}")
    print(f"result:{indexes[predicted_class.item()]}")
    return indexes[predicted_class.item()]
