import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Dataset import EnvDataset
from Param import *
import numpy as np
from torchvision import models
import random
from Dataset import EnvDataset, EnvDataset_d
from arg_parser import parse_arguments
#from ceiling import ConvNet
from torchvision import models
#输入是两张图片，输出其中一张图片的删除的物体类别
#两张图片映射的向量差值作为分类器的输入
class cf(nn.Module):
    def __init__(self):
        super(cf,self).__init__()
        self.emb=emb_A()
        self.fc=nn.Sequential(
            nn.Linear(Param.room_emb_size,32),
            nn.ReLU(),
            nn.Linear(32,3)
        )
        self.softmax=nn.Softmax()
    def forward(self,cur_obs_info,cur_obs_d_info):
        emb=self.emb(cur_obs_info,cur_obs_d_info)
        res = self.fc(emb)
        #pre=torch.argmax(scores,dim=1)
        return res
class CustomResNet(nn.Module):
    def __init__(self, output_dim=50):
        super(CustomResNet, self).__init__()
        # Load a pre-trained ResNet model
        self.resnet = models.resnet18()
        # 加载本地模型权重
        state_dict = torch.load('resnet18-f37072fd.pth')
        self.resnet.load_state_dict(state_dict)
        # Modify the first convolutional layer to accept 8x8x3 input
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Adjust the max pooling layer to work with smaller input
        #self.resnet.maxpool = nn.Identity()  # Remove max pooling layer
        
        # Replace the fully connected layer with a new one
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_dim)
    
    def forward(self, env_info):
        if len(env_info.shape)==4: env_info=env_info.unsqueeze(1)
        x=env_info      
        x= x.reshape(x.shape[0]*x.shape[1],x.shape[2],x.shape[3],x.shape[4])
        x=x.transpose(1,3).transpose(2,3)
        x = self.resnet(x)
        x=x.reshape(env_info.shape[0],env_info.shape[1],Param.room_emb_size)
        #results=x[np.arange(tgt_rooms.shape[0]), tgt_rooms, :]
        return x
class emb_A(nn.Module):
    def __init__(self):
        super(emb_A,self).__init__()
        self.embedding=CustomResNet(Param.room_emb_size)
        
    def forward(self,cur_obs_info,cur_obs_d_info):
        room_embs_A = self.embedding(cur_obs_info)
        room_d_embs_A = self.embedding(cur_obs_d_info)
        diff_emb=(room_embs_A-room_d_embs_A).squeeze()
        #diff_emb = torch.cat([room_embs_A,room_d_embs_A],dim=-1).squeeze()
        #outputs=self.fc(diff_emb)
        return diff_emb
args = parse_arguments()
#预训练好的图像向量直接分类
def guess_room_train():
    # 加载训练数据集
    train_dataset = EnvDataset_d(Param.obs_train_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    task=cf()
 
    optimizer = optim.Adam(task.parameters(), lr=Param.lr_task, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    # 训练agent A和agent B

    #用早停方法，防止过拟合
    best_val_acc = 0.0  # 记录最佳验证集准确率
    best_val_loss = 100000
    patience = 100  # 设置耐心值，即连续多少个训练周期验证集准确率没有提高时停止训练
    counter = 0  # 用于计数连续没有提高的训练周期

    for i in range(Param.epoch):
        tgt = []
        pred = []
        total_loss = 0
        
        for data,data_d,label in train_dataloader: 
            tgt.append(label)
            cur_obs_info = data.to(torch.float32)
            cur_obs_d_info = data_d.to(torch.float32)
    
            # agent A的前向传播
            outputs= task(cur_obs_info,cur_obs_d_info)
            
            #每次都要梯度清零，否则之前的梯度会累积，影响模型的收敛性
            optimizer.zero_grad()
            loss= criterion(outputs, label)
            
            total_loss+=loss.item()
            _, pre = torch.max(outputs, 1)
            pred.append(pre)
            
            # 反向传播和优化
            
            loss.backward()
            optimizer.step()         
                     
        tgt = np.concatenate(tgt, axis=0)
        pred = np.concatenate(pred, axis=0)
                
        task.eval()
        acc_train = np.mean(tgt == pred)
        total_loss=total_loss/(len(train_dataset)/args.batch_size)
        with open("results/Classify_train_acc_rn_no.txt",'a') as fp:
            fp.write(str(acc_train)+'\n')
        with open("results/Classify_train_loss_rn_no.txt",'a') as fp:
            fp.write(str(total_loss)+'\n')
            
        print("epoch{}: \nacc = {}, loss = {}".format(i, acc_train, total_loss))
        #print("epoch{}: \nacc = {}".format(i, np.mean(accum_tgt == accum_pred)))
        
        acc_eval,total_loss_eval=guess_room_evaluate(task,criterion)
        
        # 检查验证集准确率是否提高
        if acc_eval > best_val_acc:
            best_val_acc = acc_eval
            torch.save(task.state_dict(), 'Classify_rn_no.pth')
            counter = 0
        else:
            counter += 1
        '''
        if total_loss_eval < best_val_loss:
            best_val_loss=total_loss_eval
            torch.save(task.state_dict(), 'Classify.pth')
        '''
        '''
        # 检查耐心值，如果连续多个周期准确率没有提高，则停止训练
        if counter >= patience or acc_eval==1.0:
            print("Training stopped due to early stopping.")
            
            break
        ''' 
    
            #guess_room_evaluate(conv_net)
        #print(f"Epoch {i+1}/{num_epochs}, Agent A Loss: {running_loss_a/len(train_dataloader)}, Agent B Loss: {running_loss_b/len(train_dataloader)}, Accuracy: {100*correct/total}%")

    print("训练完成")

def guess_room_evaluate(task,criterion):
    # 使用训练好的模型进行预测和计算准确率
    eval_dataset = EnvDataset_d(Param.obs_eval_dir) # 请提供测试数据集
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)
    task.eval()
    
    tgt = []
    pred = []
    total_loss=0
    
    with torch.no_grad():
        for data,data_d,label in eval_dataloader: 
            tgt.append(label)
            cur_obs_info = data.to(torch.float32)
            cur_obs_d_info = data_d.to(torch.float32)

            outputs= task(cur_obs_info,cur_obs_d_info)  
            loss=  criterion(outputs, label)
            _, pre = torch.max(outputs, 1)   
            pred.append(pre)
            total_loss+=loss.item()

    tgt = np.concatenate(tgt, axis=0)
    pred = np.concatenate(pred, axis=0)
    acc_eval=np.mean(tgt == pred)
    total_loss=total_loss/(len(eval_dataset)/args.batch_size)
    with open("results/Classify_eval_acc_rn_no.txt",'a') as f:
        f.write(str(acc_eval)+'\n')
    with open("results/Classify_eval_loss_rn_no.txt",'a') as fp:
        fp.write(str(total_loss)+'\n')
    print("eval acc = {}".format(acc_eval))
    return acc_eval,total_loss

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    setup_seed(args.seed)
    guess_room_train()
