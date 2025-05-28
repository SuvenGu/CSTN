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

# ## 1-8 acc_pre tmn tmx srad
mean_c=np.array([455.69,51.9398428,  169.45926927, 1877.14472879])
std_c = np.array([ 329.40,115.44607918 , 123.6412339, 718.64770224])

# pdsi, pet, srad tmn tmx 2379 10
mean0 = np.array([582.88267721,827.32131659,1770.52155654,51.9398428,169.45926927])
std0 = np.array([135.25408365, 579.12323274, 663.31150172,115.44607918,123.6412339])

    
class CropAttriMappingDataset(Dataset):
    """
    crop classification dataset
    """
    def __init__(self, path,c_dim=14,T=10,T_a=6):
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
        self.cond = x[:,100:]
        self.y = data[:,0].astype("int64")

        self.x =rearrange(self.x,"b (t c)->b t c",t = T)
        self.cond =rearrange(self.cond,"b (t c)->b t c",c =14 )
        # 计算各类植被指数
        NDVI =  (self.x[:,:,6]-self.x[:,:,2])/(self.x[:,:,6]+self.x[:,:,2]+1e-8)
        REP = (705+35*(0.5*(self.x[:,:,5]+self.x[:,:,2])-self.x[:,:,3])/(self.x[:,:,4]-self.x[:,:,3]+1e-8))/1000
        NDSVI = (self.x[:,:,8]-self.x[:,:,2])/(self.x[:,:,8]+self.x[:,:,2]+1e-8)
        NDTI =(self.x[:,:,8]-self.x[:,:,9])/(self.x[:,:,8]+self.x[:,:,9])
        EVI = 2.5* (self.x[:,:,6]-self.x[:,:,2])/( self.x[:,:,6]+6*self.x[:,:,2]-7.5*self.x[:,:,0]+1)
        LSWI = (self.x[:,:,6]-self.x[:,:,8])/(self.x[:,:,6]+self.x[:,:,8])
        # # # 加pdsi, pet, srad tmn tmx 2379 10
        ind = [2,3,7,9,10]
        self.vi = np.stack((NDVI, REP, NDSVI,NDTI,EVI,LSWI), axis=2)
        v=np.stack((NDVI, NDSVI,LSWI), axis=2)

        # 归一化
        self.x = (self.x - mean)/std

        ind = [4,9,10,7]
        # ### 计算累积降水
        cond_pre = self.cond[:,:,4]
        cond_acc = np.zeros_like(cond_pre)
        cond_acc[:,0] = cond_pre[:,0]
        for i in np.arange(1,cond_pre.shape[1]):
            cond_acc[:,i] = cond_acc[:,(i-1)]+cond_pre[:,i]
        self.cond[:,:,4] = cond_acc  
        # ###

        self.cond = self.cond[:,:,ind]
        self.cond = (self.cond - mean_c)/std_c


        self.x =rearrange(self.x,"b t c->b c t").astype("float32")
        self.cond =rearrange(self.cond,"b t c->b c t").astype("float32")
       
        # tsne可视化用，为了使得与DCM取到相同的样本
        torch.manual_seed(50)
        new_idx = torch.randperm(self.x.shape[0])
        print(new_idx)
        self.x = self.x[new_idx]
        self.y = self.y[new_idx]
        self.cond = self.cond[new_idx]
        self.vi = self.vi[new_idx]


        # ##将类别转为只有玉米、大豆和其他
        self.y[self.y==2]=0
        self.y[self.y==3]=0
        self.y[self.y==4]=2

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx],self.y[idx],self.cond[idx],self.vi[idx]


class CropAttriMappingDataset_TGT(Dataset):
    """
    crop classification dataset
    """
    def __init__(self, path,c_dim=14,T=10,T_a=6):
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
        self.cond = x[:,100:]
        self.y = data[:,0].astype("int64")

        self.x =rearrange(self.x,"b (t c)->b t c",t = T)
        self.cond =rearrange(self.cond,"b (t c)->b t c",c =14 )
        # 计算各类植被指数
        NDVI =  (self.x[:,:,6]-self.x[:,:,2])/(self.x[:,:,6]+self.x[:,:,2]+1e-8)
        REP = (705+35*(0.5*(self.x[:,:,5]+self.x[:,:,2])-self.x[:,:,3])/(self.x[:,:,4]-self.x[:,:,3]+1e-8))/1000
        NDSVI = (self.x[:,:,8]-self.x[:,:,2])/(self.x[:,:,8]+self.x[:,:,2]+1e-8)
        NDTI =(self.x[:,:,8]-self.x[:,:,9])/(self.x[:,:,8]+self.x[:,:,9])
        EVI = 2.5* (self.x[:,:,6]-self.x[:,:,2])/( self.x[:,:,6]+6*self.x[:,:,2]-7.5*self.x[:,:,0]+1)
        LSWI = (self.x[:,:,6]-self.x[:,:,8])/(self.x[:,:,6]+self.x[:,:,8])
        # # # 加pdsi, pet, srad tmn tmx 2379 10
        ind = [2,3,7,9,10]
        self.vi = np.stack((NDVI, REP, NDSVI,NDTI,EVI,LSWI), axis=2)
        v=np.stack((NDVI, NDSVI,LSWI), axis=2)

        # 归一化
        self.x = (self.x - mean)/std

        ind = [4,9,10,7]
        # ### 计算累积降水
        cond_pre = self.cond[:,:,4]
        cond_acc = np.zeros_like(cond_pre)
        cond_acc[:,0] = cond_pre[:,0]
        for i in np.arange(1,cond_pre.shape[1]):
            cond_acc[:,i] = cond_acc[:,(i-1)]+cond_pre[:,i]
        self.cond[:,:,4] = cond_acc  
        # ###

        self.cond = self.cond[:,:,ind]
        self.cond = (self.cond - mean_c)/std_c


        self.x =rearrange(self.x,"b t c->b c t").astype("float32")
        self.cond =rearrange(self.cond,"b t c->b c t").astype("float32")
       
        # tsne可视化用，为了使得与DCM取到相同的样本
        torch.manual_seed(50)
        new_idx = torch.randperm(self.x.shape[0])
        print(new_idx)
        self.x = self.x[new_idx]
        self.y = self.y[new_idx]
        self.cond = self.cond[new_idx]


        # ##将类别转为只有玉米、大豆和其他
        self.y[self.y==2]=0
        self.y[self.y==3]=0
        self.y[self.y==4]=2

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx],self.y[idx],self.cond[idx],self.vi[idx]
    

class CropAttriIMGMappingDataset(Dataset):
    """
    crop classification dataset
    """
    def __init__(self, img,label, cond,T=10,c_dim=14,T_a=6):
        self.x = img
        self.cond = cond
        self.y = label.astype("int64")
        self.x =rearrange(self.x,"b (t c)->b t c",t = T)
        self.cond =rearrange(self.cond,"b (t c)->b t c",c =14 )
        # 计算各类植被指数
        NDVI =  (self.x[:,:,6]-self.x[:,:,2])/(self.x[:,:,6]+self.x[:,:,2]+1e-6)
        REP = (705+35*(0.5*(self.x[:,:,5]+self.x[:,:,2])-self.x[:,:,3])/(self.x[:,:,4]-self.x[:,:,3]+1e-8))/1000
        NDSVI = (self.x[:,:,8]-self.x[:,:,2])/(self.x[:,:,8]+self.x[:,:,2]+1e-6)
        NDTI =(self.x[:,:,8]-self.x[:,:,9])/(self.x[:,:,8]+self.x[:,:,9]+1e-6)
        EVI = 2.5* (self.x[:,:,6]-self.x[:,:,2])/( self.x[:,:,6]+6*self.x[:,:,2]-7.5*self.x[:,:,0]+1)
        LSWI = (self.x[:,:,6]-self.x[:,:,8])/(self.x[:,:,6]+self.x[:,:,8]+1e-6)

        # # # 加pdsi, pet, srad tmn tmx 2379 10
        ind = [2,3,7,9,10]
        climate = self.cond[:,:,ind]
        climate  = (climate - mean0)/std0

        self.vi = np.stack((NDVI, REP, NDSVI,NDTI,EVI,LSWI), axis=2)
        v=np.stack((NDVI, NDSVI,LSWI), axis=2)
        self.vi = np.concatenate((climate.reshape(climate.shape[0], -1),v.reshape(climate.shape[0], -1)),1).astype("float32")

        self.x = (self.x - mean)/std

        ind = [4,9,10,7]
        # ### 计算累积降水
        cond_pre = self.cond[:,:,4]
        cond_acc = torch.zeros_like(cond_pre)
        cond_acc[:,0] = cond_pre[:,0]
        for i in np.arange(1,cond_pre.shape[1]):
            cond_acc[:,i] = cond_acc[:,(i-1)]+cond_pre[:,i]
        self.cond[:,:,4] = cond_acc  
        # ###

        self.cond = self.cond[:,:,ind]
        self.cond = (self.cond - mean_c)/std_c


        self.x =rearrange(self.x,"b t c->b c t").astype("float32")
        self.cond =rearrange(self.cond,"b t c->b c t").float()
  

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx],self.y[idx],self.cond[idx],self.vi[idx]
