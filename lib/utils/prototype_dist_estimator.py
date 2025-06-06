import os
import torch
import torch.utils.data
import torch.distributed
import torch.backends.cudnn


class prototype_dist_estimator():
    def __init__(self, feature_num, cfg,proto=None,amount=None):
        super(prototype_dist_estimator, self).__init__()

        self.cfg = cfg
        self.class_num = cfg.MODEL.NUM_CLASSES
        self.feature_num = feature_num
        # momentum 
        self.use_momentum = cfg.CONTRAST.USE_MOMENTUM
        self.momentum = cfg.CONTRAST.MOMENTUM

        # init prototype
        self.init(feature_num=feature_num, proto = proto,amount = amount)

    def init(self, feature_num, proto=None,amount=None):
        if proto:
            # if feature_num == self.cfg.MODEL.NUM_CLASSES:
            #     resume = os.path.join(resume, 'prototype_out_dist.pth')
            # elif feature_num == self.feature_num:
            #     resume = os.path.join(resume, 'prototype_feat_dist.pth')
            # else:
            #     raise RuntimeError("Feature_num not available: {}".format(feature_num))
            # print("Loading checkpoint from {}".format(resume))
            proto = torch.load(proto, map_location=torch.device('cpu'))
            amount = torch.load(amount, map_location=torch.device('cpu'))
            self.Proto = proto.cuda(non_blocking=True)
            self.Amount = amount.cuda(non_blocking=True)
        else:
            self.Proto = torch.zeros(self.class_num, feature_num).cuda(non_blocking=True)
            self.Amount = torch.zeros(self.class_num).cuda(non_blocking=True)

    def update(self, features, labels):
        mask = (labels != self.cfg.IGNORE_LABEL)
        # remove IGNORE_LABEL pixels
        labels = labels[mask]
        features = features[mask]
        if not self.use_momentum:
            N, A = features.size()
            C = self.class_num
            # refer to SDCA for fast implementation
            features = features.view(N, 1, A).expand(N, C, A)
            onehot = torch.zeros(N, C).cuda()
            onehot.scatter_(1, labels.view(-1, 1), 1)
            NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)
            features_by_sort = features.mul(NxCxA_onehot)
            Amount_CXA = NxCxA_onehot.sum(0)
            Amount_CXA[Amount_CXA == 0] = 1
            mean = features_by_sort.sum(0) / Amount_CXA
            sum_weight = onehot.sum(0).view(C, 1).expand(C, A)
            weight = sum_weight.div(
                sum_weight + self.Amount.view(C, 1).expand(C, A)
            )
            weight[sum_weight == 0] = 0
            self.Proto = (self.Proto.mul(1 - weight) + mean.mul(weight)).detach()
            self.Amount = self.Amount + onehot.sum(0)

            # print("PRO:",self.Proto.shape)
        else:
            # momentum implementation
            ids_unique = labels.unique()
            for i in ids_unique:
                i = i.item()
                mask_i = (labels == i)
                feature = features[mask_i]
                feature = torch.mean(feature, dim=0)
                self.Amount[i] += len(mask_i)
                self.Proto[i, :] = self.momentum * feature + self.Proto[i, :] * (1 - self.momentum)
        
    def save(self, name):
        torch.save({'Proto': self.Proto.cpu(),
                    'Amount': self.Amount.cpu()
                    },
                   os.path.join(self.cfg.OUTPUT_DIR, name))