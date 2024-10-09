from sklearn.model_selection import train_test_split
from SpotDiff1 import SpotDiff
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from Agents_new import *
from Dataset import EnvDataset_d,EnvDataset_all
from Param import *
import random
from sklearn.metrics import confusion_matrix
#from gr_fd import gr_fd
#from gr_fd_fc import *
from arg_parser import parse_arguments
#！！猜物体颜色！！

args = parse_arguments()

def clear_file(file_path):
    with open(file_path+"/train_acc.txt","w") as f:
        f.write('')
    with open(file_path+"/train_reward.txt","w") as f:
        f.write('')
    with open(file_path+"/train_loss_A.txt","w") as f:
        f.write('')
    with open(file_path+"/train_loss_B.txt","w") as f:
        f.write('')
    with open(file_path+"/eval_acc.txt","w") as f:
        f.write('')
    with open(file_path+"/pattern.txt","w") as f:
        f.write('')

def guess_room_train(train_dataset,eval_dataset):
    clear_file("mt2")
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    """
    train
    :return:
    """
    #train_keyset = EnvDataset_all(Param.key_train_dir)
    #train_keyloader = DataLoader(train_keyset, batch_size=args.batch_size)
    #train_ballset = EnvDataset_all(Param.ball_train_dir)
    #train_ballloader = DataLoader(train_ballset, batch_size=args.batch_size)
    #train_boxset = EnvDataset_all(Param.box_train_dir)
    #train_boxloader = DataLoader(train_boxset, batch_size=args.batch_size)
    #type_obs_info=np.concatenate((keyset, ballset, boxset))
    #print(type_obs_info)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    #task = gr_fd().to(device)
    task = SpotDiff().to(device)
    # if Param.is_gpu: task = task.to(Param.gpu_device)
    opt = Adam(task.parameters(), lr=Param.lr_task, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)
    # opt = SGD(task.parameters(), lr=Param.lr, momentum=0.9)
    accum_tgt = []; accum_pred = []
    best_eval=0
    for i in range(150):
        tgt = []
        pred_type = []
        cur_sent = None
        total_loss_A = 0; total_loss_B = 0
        total_reward=0
        
        #for  data_batch,key,ball,box in zip(train_dataloader,train_keyloader,train_ballloader,train_boxloader):
        for data,data_d,label in train_dataloader: 
            opt.zero_grad()
            #data,data_d,label=data_batch
            tgt.append(label)
            #data,data_d, label,key,ball,box = data.to(device), data_d.to(device),label.to(device),key.to(device),ball.to(device),box.to(device)
    
            cur_obs_info = data.to(torch.float32).to(device)
            cur_obs_d_info = data_d.to(torch.float32).to(device)
            #key=key.to(torch.float32)
            #ball=ball.to(torch.float32)
            #box=box.to(torch.float32)
            #print(step)
            task.train()
            task.agentA.train()
            task.agentB.train()
            # --- FORWARD ----
            # num_room = num_room.to(torcht.float32); num_obs = num_obs.to(torch.float32)
            # cur_env_info = cur_env_info.to(torch.float32)
            
            #type_idxes, token_probs,type_probs , sent = task(cur_obs_info,cur_obs_d_info,key, ball, box)
            type_idxes, token_probs,type_probs , sent = task(cur_obs_info,cur_obs_d_info, guess_attribute="color")
            
            pred_type.append(type_idxes.cpu().numpy())
            # --- BACKWARD ---
            reward_type = np.ones_like(type_idxes.cpu().numpy())
            reward_type[type_idxes.cpu().numpy() != label.cpu().numpy() ] = -1
            #reward *= Param.reward
            #positive_numbers = [x for x in reward if x > 0]
            #print(positive_numbers)
            cur_loss_A, cur_loss_B = task.backward(token_probs, type_probs[0], torch.Tensor(reward_type).to(device))
            total_loss_A += cur_loss_A.item(); total_loss_B += cur_loss_B.item()
            total_reward+=sum(reward_type)
            
            opt.step()
        tgt = np.concatenate(tgt, axis=0)
        pred = np.concatenate(pred_type, axis=0)
        accum_tgt.append(tgt)
        accum_pred.append(pred)
        print("|",end='',flush=True)
        #不用10个epoch输出一次，因为很早就收敛了，如果10个epoch输出一次波动很大
        task.eval()
        task.agentA.eval()
        task.agentB.eval()
        accum_pred = np.concatenate(accum_pred, axis=0)
        accum_tgt = np.concatenate(accum_tgt, axis=0)
        
        acc_train=np.mean(accum_tgt == accum_pred)
        with open("mt2/train_acc.txt",'a') as fp:
            fp.write(str(acc_train)+'\n')
        with open("mt2/train_reward.txt",'a') as fp:
            fp.write(str(total_reward)+'\n')
        with open("mt2/train_loss_A.txt",'a') as fp:
            fp.write(str(total_loss_A)+'\n')  
        with open("mt2/train_loss_B.txt",'a') as fp:
            fp.write(str(total_loss_B)+'\n')  
        print()
        print("epoch{}: \nacc = {}, loss A = {}, loss B = {}, reward={}".format(i, acc_train, total_loss_A, total_loss_B,total_reward),flush=True) 
        #print("epoch{}: \nacc = {}, loss A = {}, loss B = {}".format(i, np.mean(accum_tgt == accum_pred), total_loss_A, total_loss_B))           
        accum_pred = []; accum_tgt = []
        acc_eval=guess_room_evaluate(eval_dataset,task)
        if acc_eval>best_eval:
            best_eval=acc_eval
            torch.save(task.state_dict(), "mt2/model.pth")


def guess_room_evaluate(eval_dataset,model):
    """
    evaluation
    :param model:
    :return:
    """
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    #eval_keyset = EnvDataset_all(Param.key_eval_dir)
    #eval_keyloader = DataLoader(eval_keyset, batch_size=args.batch_size)
    #eval_ballset = EnvDataset_all(Param.ball_eval_dir)
    #eval_ballloader = DataLoader(eval_ballset, batch_size=args.batch_size)
    #eval_boxset = EnvDataset_all(Param.box_eval_dir)
    #eval_boxloader = DataLoader(eval_boxset, batch_size=args.batch_size)
    
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)
    model.to(device).eval()
    
    tgt = []; pred = []
    #total_loss=0
    with torch.no_grad():
        
        acc_eval=[]
        #for data_batch,key,ball,box in zip(eval_dataloader,eval_keyloader,eval_ballloader,eval_boxloader): 
        for data,data_d,label in eval_dataloader:
            #data,data_d,label=data_batch
            tgt.append(label)
            #data,data_d, label,key,ball,box = data.to(device), data_d.to(device), label.to(device),key.to(device),ball.to(device),box.to(device)
            cur_obs_info = data.to(torch.float32).to(device)
            cur_obs_d_info = data_d.to(torch.float32).to(device)
            #key=key.to(torch.float32)
            #ball=ball.to(torch.float32)
            #box=box.to(torch.float32)
            type_idxes, token_probs,type_probs , sent = model(cur_obs_info,cur_obs_d_info,  choose_method="greedy",guess_attribute="color")
            for i in range(5,10):
                print(sent[i])
                print(type_idxes[i])
                print(label[i])
                print("-----")
            pred.append(type_idxes.cpu().numpy())
        tgt = np.concatenate(tgt, axis=0)
        pred = np.concatenate(pred, axis=0)
        acc_eval=np.mean(tgt == pred)
        #total_loss=total_loss/(len(eval_dataset)/args.batch_size)
        with open("mt2/eval_acc.txt",'a') as f:
            f.write(str(acc_eval)+'\n')
        #with open("fd_b/eval_loss.txt",'a') as f:
            #f.write(str(total_loss)+'\n')
        print("eval acc = {}".format(acc_eval))
        return acc_eval
def task_test(test_dataset):
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    model=SpotDiff().to(device)
    model.load_state_dict(torch.load("mt2/model.pth"))
    model.eval()
    
    tgt = []; pred = []; sents=[]
    with torch.no_grad():
        
        acc_test=[]
        for data,data_d,label in test_dataloader:
            tgt.append(label)
            cur_obs_info = data.to(torch.float32).to(device)
            cur_obs_d_info = data_d.to(torch.float32).to(device)
            
            type_idxes, token_probs,type_probs , sent = model(cur_obs_info,cur_obs_d_info,  choose_method="greedy", guess_attribute="color")
            pred.append(type_idxes.cpu().numpy())
            sents.append(sent.cpu().numpy())
        tgt = np.concatenate(tgt, axis=0)
        pred = np.concatenate(pred, axis=0)
        sents = np.concatenate(sents, axis=0)
        s={}
        d=[]
        step=0
        for i in sents:
            i=tuple(i)  
            d.append(i)      
            if i in s:
                s[i]+=1
            else:
                s[i]=1
        
        for message in s:
            label1=[0,0,0,0,0,0];label2=[0,0,0,0,0,0]
            for i in range(len(d)):
                if d[i]==message:
                    label1[tgt[i]]+=1
                    label2[pred[i]]+=1
            with open("mt2/pattern.txt",'a') as f:
                f.write(str(message)+':'+str(label1)+str(label2)+'\n')
        conf_label12 = confusion_matrix(tgt, pred)
        print(conf_label12)
        
        acc_test=np.mean(tgt == pred)
        print("test acc = {}".format(acc_test))
        return acc_test

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    setup_seed(args.seed)
    dataset=EnvDataset_d("Data/mt2/data")
    train_dataset, X_temp,= train_test_split(dataset, test_size=0.2)
    eval_dataset, test_dataset = train_test_split(X_temp, test_size=0.5)
    guess_room_train(train_dataset,eval_dataset)
    task_test(test_dataset)

