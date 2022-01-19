import sys

from torchsummary import summary

from model.resnet import resnet18

sys.path.append('./model/sourcemodel')
sys.path.append('../Cascade_Transfer_Learning/model/sourcemodel')
import torch
import torch.nn as nn
import os
import Build_Network


def freeze(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
        for param in m.parameters():
            param.requires_grad = False


def weight_reset(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
        for para in m.parameters():
            para.requires_grad = True
        m.reset_parameters()


def reset_all_parameters_dorisnet(net):
    for child in net.children():
        print('child', child)
        if isinstance(child, Build_Network.non_first_layer_cascade_Net):
            for _, c in enumerate(child.children()):
                print('c', c)
                weight_reset(c)
        else:
            weight_reset(child)


def load_conv(NetAddress, n_class, layer_index=11):
    model = torch.load(os.path.join(NetAddress, f'layer {layer_index}/trained model'),
                       map_location='cpu').module
    if layer_index == 0:
        for id, child in enumerate(model.children()):
            if id < 3:
                freeze(child)
            else:
                weight_reset(child)

        model._modules['fc3'] = nn.Linear(256, n_class)
    else:
        target_index = layer_index * 3
        for id, child in enumerate(model._modules[str(target_index)].children()):
            if id < 3:
                freeze(child)
            else:
                weight_reset(child)
        model._modules[str(target_index)]._modules['fc3'] = nn.Linear(256, n_class)

    return model

