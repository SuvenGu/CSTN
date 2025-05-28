# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import torch

from core.evaluate import accuracy
import numpy as np
from utils.SupCon import SupConLoss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from core.evaluate import F1Score,Recall,Precision
from itertools import cycle
from utils.SupCon import SupConLoss
import torch.nn.functional as F
from utils.ProLoss import PrototypeContrastiveLoss
from utils.prototype_dist_estimator import prototype_dist_estimator

logger = logging.getLogger(__name__)

def unlabeled_weight(T1,T2,epoch,af):
        alpha = 0.0
        if epoch > T1:
            alpha = (epoch-T1) / (T2-T1)*af
            if epoch > T2:
                alpha = af
        return alpha

def train(config, source_loader,target_loader,model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict,cls_num_list=None):
    num_classes = config.MODEL.NUM_CLASSES
    class_names = config.MODEL.CLASS_NAMES
    n_epoch = config.TRAIN.END_EPOCH

    loss_f1 = config.LOSS.PRO_F1
    loss_f = config.LOSS.PRO_F

    ## 设置伪标签flag
    pesudo_flag = config.PESUDO.FLAG

    ## 设置伪标签置信度阈值
    pesudo_conf_thre=config.PESUDO.CONF_THRE

    ## 设置起始使用伪标签的epoch
    pesudo_start_T1= config.PESUDO.START_T1 
    pesudo_mid_T2= config.PESUDO.MID_T2

    ## 设置伪标签 loss_weight af
    af =  config.PESUDO.af


    ##proto
    feature_num = config.MODEL.USE_T *256
    ## 使用源域数据初始化
    feat1_estimator = prototype_dist_estimator(feature_num=feature_num, cfg=config,proto =config.PROTO1,amount=config.AMOUNT )
    feat2_estimator = prototype_dist_estimator(feature_num=feature_num, cfg=config,proto =config.PROTO2,amount=config.AMOUNT )

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    cls_losses = AverageMeter()
    feat_loss = AverageMeter()
    pes_cls_loss = AverageMeter()
    cm = torch.zeros((num_classes,num_classes))

    # 加权交叉熵
    # criterion = LogitAdjust(cls_num_list).cuda()
    CE_loss = torch.nn.CrossEntropyLoss().cuda()
    pcl_criterion = PrototypeContrastiveLoss(config)

    # switch to train mode
    model.train()

    end = time.time()
    len_dataloader = len(source_loader)
    data_source_iter = iter(source_loader)
    data_target_iter = cycle(target_loader)

    loss_pes_weight = 0
    for i in range(len_dataloader):

        data_time.update(time.time() - end)
        #target = target - 1 # Specific for imagenet

        # training model using source data
        data_source = next(data_source_iter)
        s_img, s_label =data_source #s_vi: b,t,c
        batch_size = len(s_label)

        # compute output
        output_s,out1_s,out2_s,_= model(s_img)
        s_label = s_label.cuda(non_blocking=True)

        loss_c = criterion(output_s, s_label) 
        loss = loss_c

        # training model using target data
        data_target = next(data_target_iter)
        t_img, _ = data_target
        output_t,out1_t,out2_t,_= model(t_img)

        
        # 伪标签生成和处理
        loss_pes_weight  = unlabeled_weight(pesudo_start_T1,pesudo_mid_T2,epoch,af)
        loss_feat =torch.tensor(0)
        loss_feat1 =torch.tensor(0)
        pesudo_cls_loss = torch.tensor(0)
        tar_pesudo_len = 1# 分母不为0
        if pesudo_flag and epoch>pesudo_start_T1:
            soft_out2 = F.softmax(output_t, dim=1)
            prob, pesudo_label = soft_out2.max(dim=1)
            conf_mask = (prob >  pesudo_conf_thre).float()
            if sum(conf_mask):
                pesudo_cls_loss = CE_loss(
                output_t[conf_mask == 1], pesudo_label[conf_mask == 1])
                pesudo_label_mask = pesudo_label.detach().clone()
                pesudo_label_mask[conf_mask==0]=255  #用于制作mask, 忽略255的标签  
                pesudo_label_mask[pesudo_label==0]=255  #用于制作mask, 忽略255的标签  

                #  忽略其他类别
                s_label_mask = s_label.detach().clone()
                s_label_mask[s_label_mask==0]=255

                loss = loss + loss_pes_weight*pesudo_cls_loss

                ### proto
                # if loss_w3!=0:
                # update feature-level statistics
                feat2_estimator.update(features=out2_t.detach(), labels=pesudo_label_mask)
                feat2_estimator.update(features=out2_s.detach(), labels=s_label_mask)

                # contrastive loss on both domains
                loss_feat = pcl_criterion(Proto=feat2_estimator.Proto.detach(),
                                        feat=out2_s,
                                        labels=s_label_mask) \
                            + pcl_criterion(Proto=feat2_estimator.Proto.detach(),
                                        feat=out2_t,
                                        labels=pesudo_label_mask)
    
                tar_pesudo_len = sum(conf_mask)
                loss =  loss + loss_f* loss_pes_weight*loss_feat
                # loss =  loss + loss_f*loss_feat



                feat1_estimator.update(features=out1_t.reshape(out1_t.shape[0],-1).detach(), labels=pesudo_label_mask)
                feat1_estimator.update(features=out1_s.reshape(out1_s.shape[0],-1).detach(), labels=s_label_mask)

                # contrastive loss on both domains
                loss_feat1 = pcl_criterion(Proto=feat1_estimator.Proto.detach(),
                                        feat=out1_s.reshape(out1_s.shape[0],-1),
                                        labels=s_label_mask) \
                            + pcl_criterion(Proto=feat1_estimator.Proto.detach(),
                                        feat=out1_t.reshape(out1_t.shape[0],-1),
                                        labels=pesudo_label_mask)
    
                loss = loss + loss_f1* loss_pes_weight*loss_feat1


        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), s_img.size(0)) 
        cls_losses.update(loss_c.item(), s_img.size(0))
        pes_cls_loss.update(pesudo_cls_loss, tar_pesudo_len)
        feat_loss.update((loss_feat+loss_feat1).item(),s_img.size(0)*2) 

        prec1 = accuracy(output_s, s_label)
        top1.update(prec1[0].item(), s_img.size(0))

        output_s = output_s.argmax(dim=1)
        cm = cm+confusion_matrix(s_label.cpu(),output_s.cpu(),labels=[0,1,2])
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if i % config.PRINT_FREQ == 0:

            recalls = Recall(cm)
            pres = Precision(cm)
            f1s = F1Score(cm)

            # 计算有效类别
            idx = torch.nonzero(cm.sum(dim=1)).squeeze()

            m_f1 = (f1s[idx]).mean()
            m_pre = (pres[idx]).mean()
            m_rec  = (recalls[idx]).mean()
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                   'CLS_Loss {cls_loss.val:.5f} ({cls_loss.avg:.5f})\t' \
                    'Feat_Loss {feat_loss.val:.5f} ({feat_loss.avg:.5f})\t' \
                   'Pesudo_cls_loss {pes_cls_loss.val:.5f} ({pes_cls_loss.avg:.5f})\t' \
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t' \
                'F1 {avg_f1:.3f}\t'\
                'Precision {avg_precision:.3f}\t'\
                  'Recall {avg_recall:.3f}\t'\
                    .format(
                      epoch, i, len_dataloader, batch_time=batch_time,
                      speed=s_img.size(0)/batch_time.val,
                      data_time=data_time, loss=losses,cls_loss=cls_losses,feat_loss=feat_loss, pes_cls_loss=pes_cls_loss,top1=top1,avg_f1= m_f1,avg_precision = m_pre, avg_recall = m_rec)
            logger.info(msg)

            msg = '====' \
            'F1 {f1s}\t' \
            'Precision {precisions}\t' \
            'Recall {recalls}\t'.format(
            f1s=' '.join([f'{class_names[i]}:{f1s[i]:.4f}' for i in idx]),
            precisions=' '.join([f'{class_names[i]}:{pres[i]:.4f}' for i in idx]),
            recalls=' '.join([f'{class_names[i]}:{recalls[i]:.4f}' for i in idx]),
        ) 
            logger.info(msg)


            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer.add_scalar('train_top1', top1.val, global_steps)
                writer.add_scalar('train_F1', m_f1, global_steps)
                writer.add_scalar('cls_loss', cls_losses.val, global_steps)
                writer.add_scalar('feat_loss', feat_loss.val, global_steps)
                writer.add_scalar('pes_cls_loss', pes_cls_loss.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1




def validate(config, val_loader, model, criterion, output_dir, tb_log_dir,
             writer_dict=None):
    num_classes = config.MODEL.NUM_CLASSES
    class_names = config.MODEL.CLASS_NAMES
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    cm = torch.zeros((num_classes,num_classes))

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target, cond,vi) in enumerate(val_loader):
            # compute output

            output,_,_,_,_= model(input,vi[:,:,0:1])

            target = target.cuda(non_blocking=True)

            loss = criterion(output, target)
            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            prec1 = accuracy(output, target)
            top1.update(prec1[0].item(), input.size(0))

            output = output.argmax(dim=1)
            cm = cm+confusion_matrix(target.cpu(),output.cpu(),labels=[0,1,2])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        msg = cm
        logger.info(msg)
        recalls = Recall(cm)
        pres = Precision(cm)
        f1s = F1Score(cm)

        # 计算有效类别
        idx = torch.nonzero(cm.sum(dim=1)).squeeze()

        m_f1 = (f1s[idx]).mean()
        m_pre = (pres[idx]).mean()
        m_rec  = (recalls[idx]).mean()
        msg1 = '******************' \
            'Test: Time {batch_time.avg:.3f}\t' \
              'Loss {loss.avg:.4f}\t' \
              'Accuracy {top1.avg:.4f}\t' \
                'F1 {avg_f1:.4f}\t'\
                'Precision {avg_precision:.4f}\t'\
                  'Recall {avg_recall:.4f}\t'\
                    .format(
                  batch_time=batch_time, loss=losses, top1=top1,  avg_f1= m_f1,avg_precision = m_pre, avg_recall = m_rec)
        logger.info(msg1)

        msg2 = '====' \
        'F1 {f1s}\t' \
        'Precision {precisions}\t' \
        'Recall {recalls}\t'\
         '******************'\
            .format(
            f1s=' '.join([f'{class_names[i]}:{f1s[i]:.4f}' for i in idx]),
            precisions=' '.join([f'{class_names[i]}:{pres[i]:.4f}' for i in idx]),
            recalls=' '.join([f'{class_names[i]}:{recalls[i]:.4f}' for i in idx]),
        ) 
        logger.info(msg2)


        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_acc', top1.avg, global_steps)
            writer.add_scalar('valid_F1', m_f1, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    return m_f1,msg1,msg2

def predict(config, val_loader, model,cm):
    num_classes = config.MODEL.NUM_CLASSES
    class_names = config.MODEL.CLASS_NAMES
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    yt_pred_batch_list = []

    with torch.no_grad():
        end = time.time()
        for i, (input, target, cond,vi) in enumerate(val_loader):
            # compute output

            output,_,_,_,_= model(input,vi[:,:,0:1])

            target = target.cuda(non_blocking=True)

            # loss = criterion(output, target)
            # measure accuracy and record loss
            # losses.update(loss.item(), input.size(0))
            # prec1 = accuracy(output, target)
            # top1.update(prec1[0].item(), input.size(0))

            output = output.argmax(dim=1)
            cm = cm+confusion_matrix(target.cpu(),output.cpu(),labels=[0,1,2])
            yt_pred_batch_list.append(output)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        y_pred = torch.cat(
            yt_pred_batch_list, dim=0
        ).cpu().numpy()  

        # msg = cm
        # logger.info(msg)
        # print(cm)
        # recalls = Recall(cm)
        # pres = Precision(cm)
        # f1s = F1Score(cm)

        # # 计算有效类别
        # idx = torch.nonzero(cm.sum(dim=1)).squeeze()
        # m_f1 = (f1s[idx]).mean()
        # m_pre = (pres[idx]).mean()
        # m_rec  = (recalls[idx]).mean()
        # msg = '******************' \
        #     'Test: Time {batch_time.avg:.3f}\t' \
        #       'Loss {loss.avg:.4f}\t' \
        #       'Accuracy {top1.avg:.4f}\t' \
        #         'F1 {avg_f1:.4f}\t'\
        #         'Precision {avg_precision:.4f}\t'\
        #           'Recall {avg_recall:.4f}\t'\
        #             .format(
        #           batch_time=batch_time, loss=losses, top1=top1,  avg_f1= m_f1,avg_precision = m_pre, avg_recall = m_rec)
        # logger.info(msg)

        # msg = '====' \
        # 'F1 {f1s}\t' \
        # 'Precision {precisions}\t' \
        # 'Recall {recalls}\t'\
        #  '******************'\
        #     .format(
        #     f1s=' '.join([f'{class_names[i]}:{f1s[i]:.4f}' for i in idx]),
        #     precisions=' '.join([f'{class_names[i]}:{pres[i]:.4f}' for i in idx]),
        #     recalls=' '.join([f'{class_names[i]}:{recalls[i]:.4f}' for i in idx]),
        # ) 
        # logger.info(msg)
        # print(msg)


    return y_pred,cm


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
