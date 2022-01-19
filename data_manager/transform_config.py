import torchvision.transforms as transforms
import torch
from skimage.util import random_noise

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.cat((x,) * 3, axis=0)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
transform_val = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.cat((x,) * 3, axis=0)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

imagenet_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
imagenet_val = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class AddRandomNoise(object):
    def __init__(self, type='gaussian', mean=0, var=0.01, amount=0.05):
        self.type = type
        self.mean = mean
        self.var = var
        self.amount = amount

    def __call__(self, tensor):
        img_np = tensor.numpy().transpose(1, 2, 0)
        # print('shape before adding noise:', img_np.shape)
        if self.type == 'gaussian':
            out = random_noise(img_np, self.type, mean=self.mean, var=self.var)
        elif self.type == 's&p':
            out = random_noise(img_np, self.type, amount=self.amount)
        elif self.type == 'poisson':
            out = random_noise(img_np, self.type)
        elif self.type == 'speckle':
            out = random_noise(img_np, self.type, mean=self.mean, var=self.var)

        out = torch.tensor(out.transpose(2, 0, 1))

        return out

    def __repr__(self):
        if self.type == 's&p':
            return self.__class__.__name__ + '(type={0}, amount={3})'.format(self.type, self.mean, self.var,
                                                                             self.amount)
        elif self.type in ['gaussian', 'speckle']:
            return self.__class__.__name__ + '(type={0}, mean={1}, var={2})'.format(self.type, self.mean, self.var,
                                                                                    self.amount)


idc_train = transforms.Compose([transforms.ToPILImage(),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomRotation(20),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
                                ])
idc_val = transforms.Compose([transforms.ToPILImage(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])
                              ])
idc_test = transforms.Compose([transforms.ToPILImage(),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])

kvasir_ham1000_train = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(256),
                                           transforms.Resize(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.RandomRotation(90),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                           ])
kvasir_ham1000_val = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(256),
                                         transforms.Resize(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                         ])


def add_random_noise(type, mean, var, amount):
    assert type in ['gaussian', 'speckle', 'poisson', 's&p']
    return transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(256),
                               transforms.Resize(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                               AddRandomNoise(type=type, mean=mean, var=var, amount=amount)
                               ])


def add_random_noise_idc(type, mean, var, amount):
    assert type in ['gaussian', 'speckle', 'poisson', 's&p']
    return transforms.Compose([transforms.ToPILImage(),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
                               AddRandomNoise(type=type, mean=mean, var=var, amount=amount)
                               ])

def add_random_noise_chestxray8(type, mean, var, amount):
    assert type in ['gaussian', 'speckle', 'poisson', 's&p']
    return transforms.Compose([
                                transforms.Resize(size=256),
                                transforms.CenterCrop(size=224),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: torch.cat((x,) * 3, axis=0)),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                                AddRandomNoise(type=type, mean=mean, var=var, amount=amount)
                            ])


transform_options = {
    'train': transform_train,
    'val': transform_val,
    'imagenet_train': imagenet_train,
    'imagenet_val': imagenet_val,

    'idc_train': idc_train,
    'idc_val': idc_val,
    'idc_test': idc_test,

    'kvasir_train': kvasir_ham1000_train,
    'kvasir_val': kvasir_ham1000_val,

    'ham10000_train': kvasir_ham1000_train,
    'ham10000_val': kvasir_ham1000_val,

}
