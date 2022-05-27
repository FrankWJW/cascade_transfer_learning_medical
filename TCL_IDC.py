"""
TCL on IDC dataset
"""
import os

import argparse

import numpy as np
from sklearn.metrics import confusion_matrix
from torch import optim
import torch
import torch.nn as nn
from data_manager.IDC import get_dataloader
from model.CascadeNet import load_conv
from train import train
from ray import tune
from ray.tune.trial import ExportFormat

from util import AverageMeter, accuracy, AucMeter


def load_dataset(args, config):
    if args.dataset == 'IDC':
        root = args.root_dir
        path = os.path.join(args.root_dir, 'Breast_histopathplogy_data.csv')
        bs = int(config['batch_size'])
        cur = config['currerent_fold']
        ss = config['data_percentage']
        n_split = args.n_split
        n_wok = args.num_worker
        noise = None
        mean = 0
        var = 0
        amount = 0

        dataloader = get_dataloader(root, path, bs, ss, cur, n_wok, n_split, noise, mean, var, amount,
                                    random_state=config['partition_random_state'])
        train_loader, val_loader, test_loader = dataloader.get_data_loader()

        return train_loader, val_loader, test_loader


def validate_aux(val_loader, model, criterion, print_freq):
    """One epoch validation for aux type classifier"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        labels = []
        outputs = []
        logit_out = []
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # append result:
            labels.append(target.cpu().numpy())
            outputs.append(np.argmax(output.cpu().numpy(), axis=1))
            logit_out.append(output.cpu().numpy())

            # measure accuracy and record loss
            acc1, _ = accuracy(output, target, topk=(1, 2))

            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))

            if idx % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    .format(
                    idx, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, ))

        print(' * Acc@1 {top1.avg:.3f}'
              .format(top1=top1))

        labels = np.hstack(labels)
        outputs = np.hstack(outputs)
        logit_out = np.vstack(logit_out)

        cfm = confusion_matrix(labels, outputs)

    # return top1.avg, losses.avg, f1.avg, auc_meter.avg, auc_meter.cls_auc, ece.ece_, ece.acc, ece.conf
    return {'top1 accuracy': top1.avg.item(),
            'loss': losses.avg,
            'confusion_matrix': cfm,
            'predict_logit': logit_out,
            }


def train_aux(config, args, train_loader, val_loader, test_loader, checkpoint_dir, net):
    """main training function for aux"""
    # push model to gpu first
    # https://discuss.pytorch.org/t/code-that-loads-sgd-fails-to-load-adam-state-to-gpu/61783
    if torch.cuda.is_available():
        net = net.cuda()
        print("Running on GPU? --> ", all(p.is_cuda for p in net.parameters()))

    # optimizer
    lr_init = config['lr']
    optimizer = optim.Adam(net.parameters(), lr=lr_init, weight_decay=1e-5)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # learning rate scheduler
    lrscheduler = None

    # load from checkpoint
    init_step = 1
    if checkpoint_dir:
        print("Loading from checkpoint.")
        path = os.path.join(checkpoint_dir, "checkpoint")
        checkpoint = torch.load(path)
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if lrscheduler:
            lrscheduler.load_state_dict(checkpoint["LRscheduler"])
        init_step = checkpoint["step"]

    # parallel
    if torch.cuda.is_available():
        if args.n_gpu > 1:
            net = nn.DataParallel(net)
        net = net.cuda()
        print("Running on GPU? --> ", all(p.is_cuda for p in net.parameters()))

    # start training
    for step in range(init_step, args.max_iter + 1):
        print("==> training...")
        net.train()

        # scheduler
        last_lr = config['lr']
        if step != 1:
            if lrscheduler:
                lrscheduler.step()
                last_lr = lrscheduler.get_last_lr()[0]
            else:
                last_lr = config['lr']

        train_acc, train_loss = train(step, train_loader, net, criterion, optimizer, args.print_freq)
        result_val = validate_aux(val_loader, net, criterion,
                              print_freq=args.print_freq
                              )
        val_acc = result_val['top1 accuracy']
        val_loss = result_val['loss']

        result_test = validate_aux(test_loader, net, criterion, print_freq=args.print_freq)
        test_acc = result_test['top1 accuracy']
        predict_logit = result_test['predict_logit']

        # update current epoch
        if step % 5 == 0:
            # Every 5 steps, checkpoint our current state.
            # First get the checkpoint directory from tune.
            with tune.checkpoint_dir(step=step) as checkpoint_dir:
                # Then create a checkpoint file in this directory.
                path = os.path.join(checkpoint_dir, "checkpoint")
                # Save state to checkpoint file.
                # No need to save optimizer for SGD.
                torch.save({
                    "step": step + 1,
                    "model_state_dict": net.state_dict() if args.n_gpu == 1 else net.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "LRscheduler": lrscheduler.state_dict() if lrscheduler is not None else None,
                    "score": test_acc,
                    "predict_logit":predict_logit,
                }, path)
        step += 1

        tune.report(ACC=val_acc, lr=last_lr
                    , val_loss=val_loss, train_acc=train_acc.item(),
                    train_loss=train_loss)


def training(config, checkpoint_dir=None):
    """main training function"""
    args = config['args']
    config['layer_index'], config['model_name'] = config['layer_model']

    # dataset
    train_loader, val_loader, test_loader = load_dataset(args, config)

    # model
    net = load_conv(args.network_address, args.n_class, int(config['layer_index']))

    # summary(net, (3, 224, 224), device='cpu')
    # print(net)

    train_aux(config, args, train_loader, val_loader, test_loader, checkpoint_dir=checkpoint_dir, net=net)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ray tune
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")

    # folder
    parser.add_argument('--root_dir', type=str, default='/local/jw7u18/datasets/Breast Histopathology Images')
    parser.add_argument('--network_address', type=str,
                        default='/home/jw7u18/Cascade_Transfer_Learning/model/sourcemodel/Source Network 3')
    parser.add_argument('--save_folder', type=str, help='folder for saving checkpoints')
    parser.add_argument('--name', type=str, default='TCL_IDC', help='ray result folder name')

    # name dataset
    parser.add_argument('--dataset', type=str, default='IDC')
    parser.add_argument('--n_class', type=int, default=2)
    parser.add_argument('--n_split', type=int, default=5)

    parser.add_argument('--max_iter', type=int, default=200, help='max number of epoch')

    # others
    parser.add_argument('--print_freq', type=int, default=200, help='iterations')
    parser.add_argument('--n_gpu', default=1.0, type=float)
    parser.add_argument('--num_worker', type=int, default=8)

    args = parser.parse_args()


    def model_layer_iter():
        for model in ['TCL']:
            if model in ['TCL']:
                for l in [3,4,5,6]:
                    layer = l
                    yield layer, model

    analysis = tune.run(
        training,
        num_samples=50,
        resources_per_trial={"cpu": 4, "gpu": args.n_gpu},
        mode="max",
        export_formats=[ExportFormat.MODEL],
        checkpoint_score_attr="ACC",
        keep_checkpoints_num=1,
        config={
            "args": args,
            "lr": tune.loguniform(1e-5, 1e-1),
            "batch_size": tune.choice([8, 16, 32, 64]),
            "currerent_fold": tune.choice([0, 1, 2, 3, 4]),
            "data_percentage": tune.sample_from(lambda spec: spec.config.training_size/219388),
            "training_size": tune.grid_search([400, 600, 800]),
            "partition_random_state": tune.choice([0, 1, 2]),  # partitioning dataset
            "layer_model": tune.grid_search(list(model_layer_iter()))
        },
        name=args.name,
        resume=False,
        local_dir='./'
    )
