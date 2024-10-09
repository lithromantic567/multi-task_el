import torch
#增加eval数据集大小

class Param(object):
    threshold=0.6
    key_train_dir="./Data/key"
    ball_train_dir="./Data/ball"
    box_train_dir="./Data/box"
    key_eval_dir="./Data/key_eval"
    ball_eval_dir="./Data/ball_eval"
    box_eval_dir="./Data/box_eval"
    obs18_train_dir="./Data18/obs_train5k"
    obs18_eval_dir="./Data18/obs_eval500"
    obs_train_dir="./Data/obj_train5k"
    obs_eval_dir="./Data/obs_eval200"
    obs_test_dir="./Data/obs_test200"
    cf_train_dir="./Data/cf_train2k"
    cf_eval_dir="./Data/cf_eval200"
    cf_test_dir="./Data/cf_test200"
    room_emb_size=50
    nhead=2
    patch_per_size=int(room_emb_size/nhead)
    '''
    obs_feat_in_num = 13
    # obs_feat_mid_num = 64
    obs_feat_out_num = 20
    gate_feat_in_num = 4
    # gate_feat_mid_num = 32
    gate_feat_out_num = 20
    '''
    grid_emb_size_in=3
    grid_emb_size_out=10
    
    room_num = 9
    action_num=3
    dir_num = 4
    
    obs_feat_in_num = 4
    obs_feat_out_num = 10
    gate_feat_in_num = 3
    gate_feat_out_num = 10

    # room_emb_size = obs_feat_out_num + gate_feat_out_num
    voc_emb_size = 20
    voc_size = 26
    #句子起始标识符sos，结束标识符eos
    sos_idx = 0
    # eos_idx = 1
    # max_sent_len = 5
    max_sent_len = 5
    # env_dir = "/home_data/yh/dataset/EnvVersion1/res_files"
    # eval_env_dir = "/home_data/yh/dataset/EnvVersion1/eval_res_files"
    env_dir = "./obs_fully_36dic"
    eval_env_dir = "./obs_eval500"
    test_env_dir = "./obs_test"
    #eval_newenv_dir = "./env_generation/env03/eval_newenv_files"#new_env(4,4)
    #eval_newenv2_dir = "./env_generation/env03/eval_newenv2_files"#(5,5)
    #eval_newenv4_dir = "./env_generation/env03/eval_newenv4_files"#(8,8)
    dynamic_env_dir = "./env_catch/env_files"
    # dynamic_eval_env_dir = "/Users/hanyu/PycharmProjects/pythonProject/workstation/env_catch/eval_env_files"

    model_dir = "./models"
    sent_dir = "./sents"

    #batch_size = 100
    # batch_size = 10
    epoch = 20000
    # max_obj_num = 8
    max_room_num = 9
    max_obs_num = 9
    max_gate_num = 4
    reward = 1
    
    max_subgate_num=2
    max_subobs_num=4
    subroom_num=4
    # NOTE reward for the final goal
    # final_reward = 20
    # each_step_penalty = 10
    # lr_A = 0.001
    # lr_B = 0.001
    lr_task = 0.001
    # TODO
    #room_emb_size_in = obs_feat_in_num * max_obs_num + gate_feat_in_num * max_gate_num
    #10*4+10*2=60
    subroom_emb_size_in = obs_feat_out_num * max_subobs_num + gate_feat_out_num * max_subgate_num 
    room_emb_size_in = obs_feat_out_num * max_obs_num + gate_feat_out_num * max_gate_num
    # room_emb_size = obs_feat_out_num * max_obs_num + gate_feat_out_num * max_gate_num
    #room_emb_size = 32
    # room_emb_size = obs_feat_out_num + gate_feat_out_num
    # NOTE if move too much, then this task could fail
    max_move_len = 6
    route_num = 3   # the num of sampled route in calculating the node emb with structure info
    end_of_route = -1
    route_encoder_hidden_size = 50
    gpu_device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    is_gpu = True
    debug_mode = False
    is_dynamic_data = False
    #dynamic_datasize = 100
    
