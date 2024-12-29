import torchvision
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import os
import sys
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

# 超参数
echo=50
batch_size = 124
learning_rate = 0.005
momentum = 0.5
# 下载MNIST数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False,transform=torchvision.transforms.ToTensor(),download=True)
train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle =True)
test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle =True)
#定义网络结构
class NN(nn.Module):
    def __init__(self):
        super(NN,self).__init__()
        self.conv1 =nn.Sequential(
            nn.Conv2d(1, 10, 5, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 =nn.Sequential(
            nn.Conv2d(10, 20, 5, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten =nn.Flatten(1,-1)
        self.fc=nn.Sequential(
            nn.Linear(320,50),
            nn.Linear(50, 10),
        )
    def forward (self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
# 定义计算环境
device  =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter("./logs")

#注册函数
module =NN().to(device)
#损失函数
loss=nn.CrossEntropyLoss()  # 交叉熵损失
#优化器
optimizer = torch.optim.SGD(module.parameters(), lr=learning_rate)  # lr学习率
#评估
def calculate_accuracy(model, dataloader, device):
    model.eval()  # 将模型设置为评估模式
    correct_acc = 0
    total_acc = 0
    with torch.no_grad():  # 在评估模式下，不需要计算梯度
        for data in dataloader:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total_acc += targets.size(0)
            correct_acc += (predicted == targets).sum().item()
    accuracy = 100 * correct_acc / total_acc
    return accuracy
#训练
def train(echo):
    module.train()
    print("训练开始")
    #权重文件储存路径
    save_path = os.path.join(os.getcwd(),"result/weights")
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for i in range(echo):
        running_loss=0
        step = 0
        correct_acc = 0
        total_acc = 0
        train_bar = tqdm((train_dataloader), total=len(train_dataloader) ,file =sys.stdout,ncols=100)
        for data in train_bar:
            # print(data)
            data_length = len(train_dataloader)
            imgs,targets = data
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = module(imgs)
            result_loss=loss(outputs,targets)
            result_loss.backward()
            optimizer.step()
            running_loss = running_loss + result_loss
            # print(imgs.size())
            # print(outputs.size())
            # 把运行中的准确率acc算出来
            _, predicted = torch.max(outputs.data, 1)
            total_acc += targets.size(0)
            correct_acc += (predicted == targets).sum().item()
            # writer.add_images("input",imgs,step)
            # writer.add_images("output",output,step)
            writer.add_scalar("epoch:[{}/{}] loss".format(i,echo),result_loss.item(),step)
            step= step +1
            train_bar.desc = "train epoch[{}/{}]  loss:{:.3f} acc:{:.3f}".format(i,echo, running_loss,100 * correct_acc / total_acc)
            # if step % data_length-1 ==0:
                # print("当前轮次：{} run_loss:{} 准确率为：{}".format(i,running_loss,100 * correct_acc / total_acc))
        writer.close()

    print("训练结束")
    torch.save(module.state_dict(), os.path.join(save_path, "nn.pth"))


if __name__ == '__main__':
    train(echo)
    acc=calculate_accuracy(module, test_dataloader, device)
    print("模型最终准确率：{}".format(acc))
