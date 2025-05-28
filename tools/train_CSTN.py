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
import pprint
import shutil
import sys

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
import models
from config import config
from config import update_config
from core.function_crop_CDA import train
from core.function_crop_CDA import validate
# from utils.modelsummary import get_model_summary
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger

from dataset import CropAttriMappingDataset
from torch.nn import Module

def count_parameters(model: Module):
    # for p in model.parameters():
    #     if "kernel_a" in p:
    return sum(p.numel() for p in model.parameters())

torch.manual_seed(1173)

def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='data')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='')

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_cls_net')(
        config)

    # dump_input = torch.rand(
    #     (1, config.MODEL.IN_CHANNEL+config.MODEL.C_DIM,config.MODEL.T)
    # )
    # logger.info(get_model_summary(model, dump_input))

    # copy model file
    this_dir = os.path.dirname(__file__)
    models_dst_dir = os.path.join(final_output_dir, 'models')
    if os.path.exists(models_dst_dir):
        shutil.rmtree(models_dst_dir)
    shutil.copytree(os.path.join(this_dir, '../lib/models'), models_dst_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    gpus = list(config.GPUS)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # 计算模型的参数数量  耗时
    num_parameters = count_parameters(model)
    print(f"parameters: {num_parameters}")

    # for _,param in enumerate(model.state_dict()):
    #     print(param)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()

    optimizer = get_optimizer(config, model)

    best_perf = 0.0
    best_model = False
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            best_perf = checkpoint['perf']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))
            best_model = True
            
    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
            last_epoch-1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
            last_epoch-1
        )

    # Data loading code
    sourcedir =[]
    for i in config.DATASET.SOURCE_DATASET:
        sourcedir.append(os.path.join(args.dataDir, i+"_points",config.DATASET.TRAIN_SET+"_"+i+".csv"))
    
    targetdir =[]
    for i in config.DATASET.TARGET_DATASET:
        targetdir.append(os.path.join(args.dataDir, i+"_points",config.DATASET.TRAIN_SET+"_"+i+".csv"))

    valdir =[]
    for i in config.DATASET.TEST_DATASET:
        valdir.append(os.path.join(args.dataDir, i+"_points",config.DATASET.TEST_SET+"_"+i+".csv"))

    source_dataset = CropAttriMappingDataset(sourcedir,T=config.MODEL.T)
    source_loader = torch.utils.data.DataLoader(
        source_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True,
    )

    target_dataset = CropAttriMappingDataset(targetdir,T=config.MODEL.T)
    target_loader = torch.utils.data.DataLoader(
        target_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True,
    )
    


    val_dataset = CropAttriMappingDataset(valdir,T=config.MODEL.T)
    valid_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
    )


    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        lr_scheduler.step()
        # train for one epoch
        # train(config, train_loader, model, criterion, optimizer, epoch,
        #       final_output_dir, tb_log_dir, writer_dict,train_dataset.get_cls_num_list())
        train(config, source_loader, target_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict,None)
        # evaluate on validation set
        perf_indicator,msg1,msg2 = validate(config, valid_loader, model, criterion,
                                  final_output_dir, tb_log_dir, writer_dict)

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
            best_msg1 = msg1
            best_msg2 = msg2
            logger.info("************Best Model************")
            logger.info(best_msg1)
            logger.info(best_msg2)

        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': config.MODEL.NAME,
            'state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir, filename='checkpoint.pth.tar')

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()
    logger.info("************Best Model************")
    logger.info(best_msg1)
    logger.info(best_msg2)

if __name__ == '__main__':
    main()
