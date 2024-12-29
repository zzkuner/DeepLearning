import torch
from PIL import Image
from torchvision import transforms
import os
from torch import nn
import matplotlib.pyplot as plt

# 定义网络结构
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, 5, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, 5, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten(1, -1)
        self.fc = nn.Sequential(
            nn.Linear(320, 50),
            nn.Linear(50, 10),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# 定义图像预处理流程
def load_and_preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # 调整图像大小为28x28，与MNIST数据集一致
        transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
        transforms.ToTensor(),  # 将图像转换为PyTorch张量
        transforms.Normalize((0.1307,), (0.3081,)) # 归一化，使用MNIST数据集的均值和标准差
    ])
    # 打开图像文件
    image = Image.open(image_path).convert('L')  # 转换为灰度图
    # 预处理图像
    processed_image = transform(image)
    # 添加批次维度
    processed_image = processed_image.unsqueeze(0)
    return image, processed_image

# 模型预测
def predict_image(model, image, device):
    model.eval()  # 将模型设置为评估模式
    image = image.to(device)
    with torch.no_grad():  # 不计算梯度，减少内存和计算资源消耗
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# 加载模型
def load_model(model_path, device):
    model = NN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 将模型设置为评估模式
    return model

if __name__ == '__main__':
    # 设定设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 模型路径
    model_path = '../result/weights/nn.pth'  # 替换为你的模型文件路径
    # 图像路径
    image_path = '5.jpg'  # 替换为你的图像文件路径

    # 加载模型
    model = load_model(model_path, device)

    # 加载和预处理图像
    original_image, processed_image = load_and_preprocess_image(image_path)

    # 预测图像
    predicted_label = predict_image(model, processed_image, device)
    print(f"预测的类别是: {predicted_label}")

    # 显示原始图像和预处理后的图像
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(processed_image.squeeze().numpy(), cmap='gray')
    plt.title('Processed Image')
    plt.axis('off')

    plt.show()