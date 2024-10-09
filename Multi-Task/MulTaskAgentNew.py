from Agents_new import *
#from ceiling import *
from Param import *
from Classify_resnet import *
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_attention(query, key, value):
    scores = np.dot(query, key.T)  # 计算得分
    attention_weights = softmax(scores)  # 计算注意力权重
    output = np.dot(attention_weights, value)  # 加权和
    return output
# 形状特征的 Cross-Attention 块
class ShapeCrossAttention:
    def __init__(self):
        pass

    def forward(self, A, B):
        output_A_to_B = cross_attention(A, B, B)
        #output_B_to_A = cross_attention(B, A, A)
        return output_A_to_B

# 颜色特征的 Cross-Attention 块
class ColorCrossAttention:
    def __init__(self):
        pass

    def forward(self, A, B):
        output_A_to_B = cross_attention(A, B, B)
        #output_B_to_A = cross_attention(B, A, A)
        return output_A_to_B

# Cross-Attention 模块
class CrossAttentionBlock(nn.Module):
    def __init__(self):
        super(CrossAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=Param.room_emb_size, num_heads=2)

    def forward(self, query, key):
        # 注意力计算
        attn_output, _ = self.attention(query, key, key)
        return attn_output.squeeze()

class MulTaskModel(nn.Module):
    def __init__(self, agentA=None, agentB=None):
        super(MulTaskModel, self).__init__()
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
        self.cross_shape=CrossAttentionBlock()
        self.cross_color=CrossAttentionBlock()
        
        #self.room_embedding_A = GridEmbedding()  # initial state
        #self.room_embedding_B = GridEmbedding()
        # self.room_embedding_B = self.room_embedding_A  # share

    def forward(self, cur_obs_info, cur_obs_d_info, choose_method="sample", guess_attribute="type",history_sents=None, env_ids=None, route_len=None):
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
        #diff_emb=(room_embs_A-room_d_embs_A).squeeze()
        if guess_attribute=="type":
            diff_emb=self.cross_shape(room_embs_A,room_d_embs_A)
        elif guess_attribute=="color":
            diff_emb=self.cross_color(room_embs_A,room_d_embs_A)
        #diff_emb=x
        #print(diff_emb)
        sent, token_probs = self.agentA.describe_room(diff_emb, Param.max_sent_len, choose_method)

        type_idx, type_prob = self.agentB.guess_type(sent, choose_method, guess_attribute)
        #color_idx, color_prob = self.agentB.guess_color(sent, choose_method)
        return type_idx, token_probs,type_prob , sent 

    def backward(self, token_probs1, type_prob1, reward1,token_probs2,type_prob2,reward2):
        lossA1 = self.agentA.cal_guess_type_loss(token_probs1, reward1)
        lossB1 = self.agentB.cal_guess_type_loss(type_prob1, reward1)
        lossA2 = self.agentA.cal_guess_type_loss(token_probs2, reward2)
        lossB2 = self.agentB.cal_guess_type_loss(type_prob2, reward2)
        lossA=lossA1+lossA2
        lossB=lossB1+lossB2
        lossA.backward()
        lossB.backward()
        return lossA1, lossB1,lossA2,lossB2