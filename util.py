import os
try:
    import pickle5 as pickle
except:
    import pickle

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score



class BCEWithLogitsLoss(nn.Module):
    def __init__(self, num_classes=64):
        super(BCEWithLogitsLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input, target):
        # target_onehot = torch.nn.functional.one_hot(target, num_classes=self.num_classes)
        return self.criterion(input, target)


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


class AucMeter(object):
    """Store ground truth label and predict label, until epoch then calculate auc"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.y_pred = []
        self.y_true = []
        self.count = 0

    def update(self, y_pred, y_true, n=1):
        self.y_pred.append(y_pred)
        self.y_true.append(y_true)
        self.count += n

    def auc(self, option='binary'):
        if option == 'multi_cls':
            self.y_pred = np.vstack(self.y_pred)
            self.y_true = np.hstack(self.y_true).astype(np.int64)
            self.avg = roc_auc_score(self.y_true, self.y_pred, multi_class='ovr', average='weighted')

            roc_multicls = {label: [] for label in np.unique(self.y_true)}
            for label in np.unique(self.y_true):
                binary_true = (self.y_true == label).astype(np.int)
                roc_multicls[label] = roc_auc_score(binary_true, self.y_pred[:, 1])

            self.cls_auc = [roc_multicls[i] for i in roc_multicls.keys()]
        elif option == 'binary':
            self.y_pred = np.vstack(self.y_pred)
            self.y_true = np.hstack(self.y_true).astype(np.int64)
            self.avg = roc_auc_score(self.y_true, self.y_pred[:, 1])
            self.cls_auc = [0]
        elif option == 'multi_label':
            self.y_pred = np.vstack(self.y_pred)
            self.y_true = np.vstack(self.y_true).astype(np.int64)
            self.avg = roc_auc_score(self.y_true, self.y_pred)
            self.cls_auc = roc_auc_score(self.y_true, self.y_pred, average=None)
        else:
            raise NotImplementedError


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # number correctly predicted data under topk criterion
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_data(args, data, file_name='none'):
    with open(os.path.join(args.tb_folder, file_name), 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

def load_data(dir):
    with open(dir, 'rb') as f:
        data = pickle.load(f)
    f.close()
    return data