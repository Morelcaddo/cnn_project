import torch
from model.cnn import CNN
x = torch.randn(32, 3, 224, 224)
model = CNN(num_class=4)
output = model(x)
print(output.shape)
