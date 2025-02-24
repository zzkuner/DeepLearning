import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import os
from tqdm import tqdm
from torchvision import transforms

# 超参数
epochs = 50
batch_size = 256
learning_rate = 0.01

# 自定义数据集类
class FashionMNISTDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.images = np.load(images_path).reshape(-1, 28, 28)  # 直接reshape为3D数组
        self.labels = np.load(labels_path)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = image.astype(np.float32) / 255.0  # 归一化到 [0, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为 Tensor，并归一化到 [0, 1]
    transforms.RandomHorizontalFlip(p=0.5)  # 随机水平翻转，概率为 0.5
])

# 数据路径
train_images_path = './fashion-mnist/train-images.npy'
train_labels_path = './fashion-mnist/train-labels.npy'
test_images_path = './fashion-mnist/t10k-images.npy'
test_labels_path = './fashion-mnist/t10k-labels.npy'

# 加载数据集
train_dataset = FashionMNISTDataset(train_images_path, train_labels_path, transform=transform)
test_dataset = FashionMNISTDataset(test_images_path, test_labels_path, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义神经网络结构
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 0),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten(1, -1)
        self.fc = nn.Sequential(
            nn.Linear(12*12*64, 50),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(50, 10),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# 选择设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 初始化模型、损失函数和优化器
model = NN().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 计算准确率
def calculate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# 训练模型
def train(epochs):
    model.train()
    print("训练开始")
    save_path = os.path.join(os.getcwd(), "result/weights")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for epoch in range(epochs):
        running_loss = 0
        step = 0
        correct = 0
        total = 0
        train_bar = tqdm(train_dataloader, total=len(train_dataloader), ncols=100)
        for data in train_bar:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            train_bar.set_description(f"Train Epoch [{epoch}/{epochs}] Loss: {running_loss / (step + 1):.3f} Acc: {100 * correct / total:.3f}%")
            step += 1
        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{epoch}.pth"))
    print("训练结束")

if __name__ == '__main__':
    train(epochs)
    accuracy = calculate_accuracy(model, test_dataloader, device)
    print(f"最终模型准确率：{accuracy:.2f}%")