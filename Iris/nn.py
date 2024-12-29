import os
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import iris_dataloader
import numpy as np

# 初始化神经网络

class NN(nn.Module):
    def __init__ (self,in_dim,hidden_dim1,hidden_dim2,out_dim):
        super().__init__()
        self.layer1 =nn.Linear(in_dim,hidden_dim1)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.layer3 = nn.Linear(hidden_dim2, out_dim)

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# 定义计算环境
device  =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#划分数据集

custom_dataset = iris_dataloader("data/Iris.data")
train_size = int(len(custom_dataset)*0.7)
val_size =  int(len(custom_dataset)*0.2)
test_size = len(custom_dataset) -val_size -train_size

train_dataset , val_dataset , test_dataset =torch.utils.data.random_split(custom_dataset,[train_size,val_size,test_size])

train_loader = DataLoader(train_dataset,batch_size=15,shuffle =True)
val_loader = DataLoader(val_dataset,batch_size=1,shuffle =True)
test_loader = DataLoader(test_dataset,batch_size=1,shuffle =True)

#定义推理函数，并计算返回准确率
def infer(model,dataset,device):
    model.eval()
    acc_num = 0
    with torch.no_grad():
        for data in dataset:
            datas,label =data
            outputs = model(datas.to(device))
            predict_y = torch.max(outputs,dim=1)[1]
            acc_num += torch.eq(predict_y,label.to(device)).sum().item()
    acc =acc_num /len(dataset)
    return acc

def main(lr=0.005,epochs =1000):
    model =NN(4,16,8,3).to(device)
    loss_f =nn.CrossEntropyLoss()
    pg =[p for p in model.parameters()  if p.requires_grad]
    optimizer =optim.Adam(pg,lr=lr)
    #权重文件储存路径
    save_path = os.path.join(os.getcwd(),"result/weights")
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    #开始训练
    for epoch in range(epochs):
        model.train()
        acc_num =torch.zeros(1).to(device)
        sample_num =0

        train_bar = tqdm(train_loader,file =sys.stdout,ncols=100)
        for datas in train_bar:
            data,label =datas
            label =label.squeeze(-1)
            sample_num += data.shape[0]
            optimizer.zero_grad()
            outputs=model(data.to(device))
            pred_class = torch.max(outputs,dim=1)[1]
            acc_num = torch.eq(pred_class,label.to(device)).sum()

            loss =loss_f(outputs,label.to(device))
            loss.backward()
            optimizer.step()

            train_acc =acc_num/sample_num
            train_bar.desc ="train epoch[{}/{}] loss:{:.3f}".format(epoch+1,epochs,loss)
        val_acc =infer(model,val_loader,device)
        print("train epoch[{}/{}] loss:{:.3f} train_acc:{:.3f} val_acc:{:.3f}".format(epoch+1,epochs,loss,train_acc,val_acc))
        torch.save(model.state_dict(),os.path.join(save_path,"nn.pth"))

        #清零 初始化指标
        train_acc =0.
        val_acc =0.
    print("训练结束")
    test_acc =infer(model,test_loader,device)
    print("test_acc:",test_acc)

if __name__ == "__main__":
    main()










