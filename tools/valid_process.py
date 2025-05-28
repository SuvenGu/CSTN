# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import shutil
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import _init_paths
import models
from config import config
from config import update_config
from core.function_crop_CDA import predict
from utils.utils import create_logger
from dataset import CropAttriIMGMappingDataset
from osgeo import gdal
import numpy as np
import tifffile
from sklearn.metrics import precision_recall_fscore_support
from core.evaluate import F1Score,Recall,Precision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

import datetime
def writeTiff(im_data, im_width, im_height, im_bands, im_geotrans, im_proj, path):
    if 'float32' in im_data.dtype.name:
        datatype = gdal.GDT_Float32
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Byte

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype,['COMPRESS=LZW','BIGTIFF=YES'])
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

def gdal_read(path, proj=False, gt=False):
    dataset = gdal.Open(path)
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数

    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    im_proj = dataset.GetProjection()  # 地图投影信息
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵

    del dataset
    if proj == True and gt == False:
        return im_data, im_proj
    elif proj == False and gt == True:
        return im_data, im_geotrans
    elif proj == True and gt == True:
        return im_data, im_proj, im_geotrans
    else:
        return im_data
    
# 遍历气象文件夹和图像文件夹，对每张图像每个像素进行分类
def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--labelDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--outLabelDir',
                        help='label directory',
                        type=str,
                        default='data')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='test img filepath',
                        type=str,
                        default='')

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()


    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_cls_net')(
        config)

    # logger.info(get_model_summary(model, dump_input))


    print('=> loading model from {}'.format(args.testModel))
    model.load_state_dict(torch.load((args.testModel)))

    gpus = list(config.GPUS)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()

    names = os.listdir(args.dataDir)
    cm = torch.zeros((config.MODEL.NUM_CLASSES,config.MODEL.NUM_CLASSES))
    class_names = config.MODEL.CLASS_NAMES
    
    if not os.path.exists(args.outLabelDir):
        os.mkdir(args.outLabelDir)
    
    count = 0
    current_time0 = datetime.datetime.now()
    for i in names:
        count = count+1
        print(i)
        
        img_name = os.path.join(args.dataDir,i)
        label_name = os.path.join(args.labelDir,i[:-4]+"_label.tif")
        out_path = os.path.join(args.outLabelDir,i[:-4]+"_pred.tif")

        # 如果labelDir不存在
        if args.labelDir=='':
            _,im_proj, im_geotrans  = gdal_read(img_name,True,True)
            img = tifffile.imread(img_name).astype("float32")
            row,col = np.shape(img)[:2]
            label = np.zeros((img.shape[0],img.shape[1]))
            label_img = np.zeros((img.shape[0],img.shape[1]))

        else:
            _,im_proj, im_geotrans  = gdal_read(label_name,True,True)
            # label=label.transpose(1,2,0)
            label_img = tifffile.imread(label_name)
            row,col = np.shape(label_img)[:2]
        img = tifffile.imread(img_name).astype("float32")


        
        # 取所有点
        indices_conf_100 = np.where(label>-1)
        label = label[indices_conf_100]
        img = img[indices_conf_100].astype("float32")

        if len(img)==0:
            continue

        val_dataset = CropAttriIMGMappingDataset(img,label,T=config.MODEL.T)
        valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=4096*2, #4096*16
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=True
        )
        


        # evaluate on validation set
        y_pred,cm = predict(config, valid_loader, model,cm)
        
        # print("writing img!")
        y_pred = y_pred.reshape(row,col)
        writeTiff(y_pred.astype('int'),col,row,1,im_geotrans,im_proj,out_path)
    current_time1 = datetime.datetime.now()
    print("start:",current_time0)
    print("end:",current_time1)

if __name__ == '__main__':
    main()
