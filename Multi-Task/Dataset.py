# NOTE assume that agentA is in the middle of each room
import os
import ast
import pickle
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from Param import *
import numpy as np

# TODO: check read order
class EnvDataset(Dataset):
    def __init__(self, env_dir):
        self.env_dir = env_dir
        self.env_files = os.listdir(self.env_dir)
        self.env_files = [f for f in self.env_files if not f.startswith(".")]  # remove .DS_Store
    def __getitem__(self, item):
        with open("{}/obs{}.txt".format(self.env_dir, item), 'rb') as f:
            data=np.load(file=f)
            return data
            
    def __len__(self):
        return len(self.env_files) 
class EnvDataset_cp(Dataset):
    def __init__(self, obs_d_dir):
        self.obs_d_dir = obs_d_dir
        self.obs_d_files = os.listdir(self.obs_d_dir)
        self.obs_d_files = [f for f in self.obs_d_files if not f.startswith(".")]  # remove .DS_Store
    def __getitem__(self, item):
        with open("{}/obs{}.txt".format(self.obs_d_dir, item), 'rb') as f:
            data = pickle.load(f)

            return data["data"],data["label"]
            
    def __len__(self):
        return len(self.obs_d_files) 
   
class EnvDataset_d(Dataset):
    def __init__(self, obs_d_dir):
        self.obs_d_dir = obs_d_dir
        self.obs_d_files = os.listdir(self.obs_d_dir)
        self.obs_d_files = [f for f in self.obs_d_files if not f.startswith(".")]  # remove .DS_Store
    def __getitem__(self, item):
        with open("{}/obs{}.txt".format(self.obs_d_dir, item), 'rb') as f:
            data = pickle.load(f)

            return data["data"],data["data_d"],data["label"]
            
    def __len__(self):
        return len(self.obs_d_files) 

class EnvDataset_all(Dataset):
    def __init__(self, obs_d_dir):
        self.obs_d_dir = obs_d_dir
        self.obs_d_files = os.listdir(self.obs_d_dir)
        self.obs_d_files = [f for f in self.obs_d_files if not f.startswith(".")]  # remove .DS_Store
    
    def __getitem__(self, item):
        with open("{}/{}.txt".format(self.obs_d_dir, item), 'rb') as f:
            data = pickle.load(f)
            return data
    def __len__(self):
        return len(self.obs_d_files) 

