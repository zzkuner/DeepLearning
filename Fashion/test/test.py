import torch
from PIL import Image
from torchvision import transforms
import os
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

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
            nn.Linear(12 * 12 * 64, 50),
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

# 定义图像预处理流程
def load_and_preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('L')
    processed_image = transform(image)
    processed_image = processed_image.unsqueeze(0)
    processed_image = 1 - processed_image  # 反转黑白
    return processed_image

# 模型预测
def predict_image(model, image, device, confidence_threshold=0.8):
    model.eval()
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)  # 获取预测概率
        _, predicted = torch.max(outputs, 1)
        confidence = probabilities[0, predicted].item()  # 获取预测置信度
    if confidence < confidence_threshold:
        return -1, confidence  # 返回 -1 表示不支持的类别，同时返回置信度
    return predicted.item(), confidence

# 加载模型
def load_model(model_path, device):
    model = NN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 模型路径
    model_path = '../result/weights/model_epoch_49.pth'  # 模型文件路径
    if not os.path.exists(model_path):
        print(f"Model file does not exist: {model_path}")
    model = load_model(model_path, device)

    # 文件夹路径
    categories = ['Ankle boot', 'Bag', 'Coat', 'Dress', 'Pullover', 'Sandal', 'Shirt', 'Sneaker', 'Trouser', 'T-shirt']
    root_dir = './test_images'  # 替换为包含类别文件夹的目录路径

    # 创建类别映射
    label_mapping = {
        'Ankle boot': 9,
        'Bag': 8,
        'Coat': 4,
        'Dress': 3,
        'Pullover': 2,
        'Sandal': 5,
        'Shirt': 6,
        'Sneaker': 7,
        'Trouser': 1,
        'T-shirt': 0
    }

    correct = 0
    total = 0
    confidence_threshold = 0.55  # 设置置信度阈值

    true_labels = []
    predicted_labels = []
    all_probabilities = []  # 用于存储每个类别的预测概率

    for index, category in enumerate(categories):
        folder_path = os.path.join(root_dir, category)
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(folder_path, filename)
                processed_image = load_and_preprocess_image(image_path)

                # 可视化处理后的图像
                processed_image_to_display = processed_image.squeeze(0)  # 去掉批次维度
                processed_image_to_display = processed_image_to_display.permute(1, 2, 0)  # 调整维度顺序为HxWxC
                processed_image_to_display = processed_image_to_display.numpy()  # 转换为numpy数组
                plt.imshow(processed_image_to_display, cmap='gray')  # 使用灰度图显示
                plt.title(f"Processed Image: {image_path}")
                plt.show()

                predicted_label, confidence = predict_image(model, processed_image, device, confidence_threshold)
                true_label = label_mapping[category]
                print(f'Image {image_path} - True Label: {true_label}, Predicted Label: {predicted_label}, Confidence: {confidence:.2f}')
                true_labels.append(true_label)
                predicted_labels.append(predicted_label)
                total += 1
                if predicted_label == true_label:  # 检查预测是否正确
                    correct += 1

                # 获取所有类别的预测概率
                with torch.no_grad():
                    outputs = model(processed_image)
                    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                all_probabilities.append(probabilities)

    accuracy = correct / total
    print(f'Classification accuracy: {accuracy:.2f}')

    # 计算评估指标
    precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    roc_auc = roc_auc_score(true_labels, all_probabilities, multi_class='ovr')
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'ROC AUC Score: {roc_auc:.2f}')
    print('Confusion Matrix:')
    print(conf_matrix)
# 使用测试集测试
# import torch
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# from torch import nn
# import os
#
#
# # 定义神经网络结构
# class NN(nn.Module):
#     def __init__(self):
#         super(NN, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 16, 5, 1, 2),
#             nn.BatchNorm2d(16),
#             nn.ReLU()
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(16, 32, 3, 1, 0),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 3, 1, 0),
#             nn.BatchNorm2d(64),
#             nn.MaxPool2d(2),
#         )
#         self.flatten = nn.Flatten(1, -1)
#         self.fc = nn.Sequential(
#             nn.Linear(12 * 12 * 64, 50),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(50, 10),
#         )
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.flatten(x)
#         x = self.fc(x)
#         return x
#
#
# # 模型预测
# def predict_image(model, image, device):
#     model.eval()
#     image = image.to(device)
#     with torch.no_grad():
#         outputs = model(image)
#         _, predicted = torch.max(outputs, 1)
#     return predicted.item()
#
#
# # 加载模型
# def load_model(model_path, device):
#     model = NN().to(device)
#     # 设置 weights_only=True 以提高安全性
#     model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
#     model.eval()
#     return model
#
#
# # 读取数据
# def load_data():
#     train_images = np.load('train-images.npy')
#     train_labels = np.load('train-labels.npy')
#     t10k_images = np.load('t10k-images.npy')
#     t10k_labels = np.load('t10k-labels.npy')
#     return train_images, train_labels, t10k_images, t10k_labels
#
#
# if __name__ == '__main__':
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     # 模型路径
#     model_path = '../result/weights/model_epoch_49.pth'  # 替换为你的模型文件路径
#     if not os.path.exists(model_path):
#         print(f"Model file does not exist: {model_path}")
#     model = load_model(model_path, device)
#
#     # 读取数据
#     train_images, train_labels, t10k_images, t10k_labels = load_data()
#
#     # 测试集预测
#     correct = 0
#     total = 0
#     for i in range(t10k_images.shape[0]):
#         image = t10k_images[i].reshape((28, 28))  # 调整图像形状为 28x28
#         image = image.astype(np.float32) / 255.0  # 归一化到 [0, 1]
#         label = t10k_labels[i]
#
#         # 显示图像
#         plt.imshow(image, cmap='gray')
#         plt.title(f'True Label: {label}')
#         plt.show()
#
#         # 转换为tensor并添加批次维度
#         image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
#
#         # 预测
#         predicted_label = predict_image(model, image_tensor, device)
#         total += 1
#         if predicted_label == label:
#             correct += 1
#             print(f'Image {i} is correctly predicted as category {predicted_label}')
#         else:
#             print(f'Image {i} is predicted as category {predicted_label}, but true category is {label}')
#
#     accuracy = correct / total
#     print(f'Classification accuracy on test set: {accuracy:.2f}')