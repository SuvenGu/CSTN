import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from einops import rearrange

mean=np.array([0.0592465,0.08257256,0.07940245,0.122184,0.26104507,0.32686248,0.33226496,0.35085383,0.25254872,0.16759971])
std = np.array([0.03050405, 0.03372968, 0.05003787, 0.05067676, 0.07917259,0.11817433, 0.11635097, 0.12101864, 0.09216624, 0.08916556])
    
class CropAttriMappingDataset(Dataset):
    """
    crop classification dataset
    """
    def __init__(self, path,c_dim=3,T=10):
        dfs= []
        if isinstance(path, list):
            for i in path:
                print(i)
                df = pd.read_csv(i)
                dfs.append(df)
            # 将各个DataFrame转换为NumPy数组
            arrays = [df.values for df in dfs]
            data = np.concatenate(arrays, axis=0)
        else:
            data = pd.read_csv(i)
            data = np.array(data.values)

        print("data loaded!")
        print(data.shape)
        x = data[:,1:]
        self.x = x[:,:100]
        self.y = data[:,0].astype("int64")

        self.x =rearrange(self.x,"b (t c)->b t c",t = T)

        # 归一化
        self.x = (self.x - mean)/std


        self.x =rearrange(self.x,"b t c->b c t").astype("float32")

       
        # tsne可视化用，为了使得与DCM取到相同的样本
        torch.manual_seed(50) # MN的tsne使用100，其他用50
        new_idx = torch.randperm(self.x.shape[0])

        self.x = self.x[new_idx]
        self.y = self.y[new_idx]


        # # ##将类别转为只有玉米、大豆和其他
        self.y[self.y==2]=0
        self.y[self.y==3]=0
        self.y[self.y==4]=2
        # self.y[self.y==4]=0

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx],self.y[idx]


class CropAttriIMGMappingDataset(Dataset):
    """
    crop classification dataset
    """
    def __init__(self, img,label,T=10):
        self.x = img
        self.y = label.astype("int64")
        self.x =rearrange(self.x,"b (t c)->b t c",t = T)

        self.x = (self.x - mean)/std
        self.x =rearrange(self.x,"b t c->b c t").astype("float32")
  

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx],self.y[idx]
