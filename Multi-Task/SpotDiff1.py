from Agents_new import *
#from ceiling import *
from Param import *
from Classify_resnet import *

class SpotDiff(nn.Module):
    def __init__(self, agentA=None, agentB=None):
        super(SpotDiff, self).__init__()
        self.agentA = AgentA() if agentA is None else agentA
        self.agentB = AgentB() if agentB is None else agentB
        
        # 创建ConvA和ConvB模型实例
        #self.room_embedding_A = ConvNet()
        #self.room_embedding_B = ConvNet()
        self.room_embedding_A = CustomResNet()
        self.room_embedding_B = CustomResNet()
        
        
        state_dict = torch.load('../ceiling_resnet/model.pth')
        params = {k.replace('emb.embedding.', ''): v for k, v in state_dict.items() if k.startswith('emb.embedding')}
        # 加载预训练的模型参数
        self.room_embedding_A.load_state_dict(params)
        self.room_embedding_B.load_state_dict(params)
        # 将模型设置为评估模式
        #self.room_embedding_A.eval()
        #self.room_embedding_B.eval()
        #self.room_embedding_A.requires_grad=False
        #self.room_embedding_B.requires_grad=False
        
        self.diff_fc=nn.Sequential(
            nn.Linear(Param.room_emb_size*2, Param.room_emb_size),
            nn.ReLU()
        )
            
        
        #self.room_embedding_A = GridEmbedding()  # initial state
        #self.room_embedding_B = GridEmbedding()
        # self.room_embedding_B = self.room_embedding_A  # share

    def forward(self, cur_obs_info, cur_obs_d_info, guess_attribute="type", choose_method="sample", history_sents=None, env_ids=None, route_len=None):
        #batch_size*1 50
        #tgt_types_arr = np.array(label_type).astype(int)
        #tgt_colors_arr = np.array(label_color).astype(int)
        #(50,9,3,3,3)
        #obs_info= env_info[:,:,2:5,2:5,:]
        #(50,9,50)->(50,50)
        
        room_embs_A = self.room_embedding_A(cur_obs_info)
        room_d_embs_A = self.room_embedding_A(cur_obs_d_info)
        #x = torch.cat([room_embs_A,room_d_embs_A],dim=-1).squeeze()#(20,100)    
        #print(room_embs_A)
        #print(room_d_embs_A)
        #diff_emb=self.diff_fc(x).squeeze()
        #diff_emb=(room_embs_A-room_d_embs_A).squeeze().detach()
        diff_emb=(room_embs_A-room_d_embs_A).squeeze()
        #diff_emb=x
        #print(diff_emb)
        sent, token_probs = self.agentA.describe_room(diff_emb, Param.max_sent_len, choose_method)

        type_idx, type_prob = self.agentB.guess_type(sent, choose_method,guess_attribute)
        #color_idx, color_prob = self.agentB.guess_color(sent, choose_method)
        return type_idx, token_probs,type_prob , sent 

    def backward(self, token_probs, type_prob, reward):
        lossA = self.agentA.cal_guess_type_loss(token_probs, reward)
        lossB = self.agentB.cal_guess_type_loss(type_prob, reward)
        lossA.backward()
        lossB.backward()
        return lossA, lossB