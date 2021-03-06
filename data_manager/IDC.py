import os

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from data_manager.transform_config import idc_train_config, idc_test_config


class IDC(Dataset):
    def __init__(self, root, df_data, transform=None):
        super().__init__()
        self.root = root
        self.df_data = df_data
        self.df = df_data.values
        self.class_weights = self.class_weight()
        self.sample_weights = self.sample_weight()

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path, label = self.df[index]
        img_path = os.path.join(self.root, img_path)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (224, 224))
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def class_weight(self):
        N = len(self.df_data['target'])
        value, count = np.unique(self.df_data['target'], return_counts=True)
        return N / count

    def sample_weight(self):
        weight = self.class_weights
        sample_weight = [weight[i] for i in self.df_data['target']]
        return sample_weight


class get_dataloader:
    def __init__(self, data_root, path, batch_size, subset_size, currerent_fold,
                 num_workers, n_split=5, random_state=0):
        self.ran_s = random_state
        self.trans_train = idc_train_config
        self.trans_valid = idc_test_config

        self.data = pd.read_csv(path).iloc[:, 1:]
        self.train, self.test = train_test_split(self.data, test_size=0.2
                                                 , random_state=self.ran_s)
        self.patients_train = self.train.patient_id.unique()
        self.patients_test = self.test.patient_id.unique()

        self.dataroot = data_root
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.subset_size = subset_size
        self.n_split = n_split
        self.fold = currerent_fold

        x = self.train.patient_id.unique()
        kf = KFold(n_splits=self.n_split)

        print(f'{self.n_split} cross validation, patiens nums')
        for train_id, test_id in kf.split(x):
            print('train -  {}   |   test -  {}'.format(
                len(x[train_id]), len(x[test_id])))
        self.all_fold = list(kf.split(x))

    def get_data_loader(self):
        train_id, val_id = self.all_fold[self.fold]
        train_id = self.patients_train[train_id]
        val_id = self.patients_train[val_id]
        train_p = self.data.loc[self.data.patient_id.isin(train_id), :].copy()[['path', 'target']]
        val_p = self.data.loc[self.data.patient_id.isin(val_id), :].copy()[['path', 'target']]
        test = self.test.copy()[['path', 'target']]

        if self.subset_size == 1.0:
            pass
        else:
            train_p, _ = train_test_split(train_p, test_size=1 - self.subset_size, random_state=self.ran_s)

        dataset_train = IDC(self.dataroot, df_data=train_p, transform=self.trans_train)
        dataset_valid = IDC(self.dataroot, df_data=val_p, transform=self.trans_valid)
        dataset_test = IDC(self.dataroot, df_data=test, transform=self.trans_valid)

        # handle data imbalance
        sampler = WeightedRandomSampler(weights=dataset_train.sample_weights, num_samples=len(dataset_train))

        loader_train = DataLoader(dataset=dataset_train, sampler=sampler, batch_size=self.batch_size,
                                  num_workers=self.num_workers)
        loader_valid = DataLoader(dataset=dataset_valid, batch_size=self.batch_size, shuffle=False,
                                  num_workers=self.num_workers)
        loader_test = DataLoader(dataset=dataset_test, batch_size=self.batch_size, shuffle=False,
                                 num_workers=self.num_workers)

        return loader_train, loader_valid, loader_test


if __name__ == '__main__':
    root = '/Users/juanwenwang/Datasets/Breast Histopathology Images'
    path = '/Users/juanwenwang/PycharmProjects/Cascade_Transfer_Learning/save/IDC/Breast_histopathplogy_data.csv'
    batch_size = 1
    subset_size = 1.0
    n_split = 5
    currerent_fold = 0
    num_workers = 0

    dataloader = get_dataloader(root, path, batch_size, subset_size, currerent_fold, num_workers, n_split)
    train_loader, val_loader, test_loader = dataloader.get_data_loader()
    print(iter(val_loader).__next__())
    print(len(train_loader), len(val_loader), len(test_loader))