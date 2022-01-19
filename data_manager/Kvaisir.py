from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from PIL import Image
from data_manager.transform_config import transform_options, add_random_noise


class Kvasir(Dataset):
    def __init__(self, root, df_data, transform=None):
        super().__init__()
        self.root = root
        self.df_data = df_data
        self.df = df_data.values

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path, label = self.df[index]
        img_path = os.path.join(self.root, 'kvasir-dataset', img_path)
        image = Image.open(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class get_dataloader_kvasir:
    def __init__(self, data_root, path, batch_size, subset_size, currerent_fold, num_workers, n_split=5, noise='none',
                 mean=0,var=0.01,amount=0.05):
        self.trans_train = transform_options['kvasir_train']
        self.trans_valid = transform_options['kvasir_val']
        if noise in ['gaussian', 'speckle', 's&p', 'poisson'] :
            self.trans_valid = add_random_noise(noise, mean=mean, var=var, amount=amount)
            print(self.trans_valid)
        else:
            pass

        self.data = pd.read_csv(path).iloc[:, 1:]
        self.dataroot = data_root
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.subset_size = subset_size
        self.n_split = n_split
        self.fold = currerent_fold

        x = self.data.index
        y = self.data.target
        kf = StratifiedKFold(n_splits=self.n_split)

        print(f'{self.n_split} cross validation')
        for train_id, test_id in kf.split(x, y):
            print('train -  {}   |   test -  {}'.format(
                len(x[train_id]), len(x[test_id])))
        self.all_fold = list(kf.split(x, y))

    def get_data_loader(self):
        train_id, test_id = self.all_fold[self.fold]
        train_p = self.data.loc[self.data.index.isin(train_id), :].copy()[['path', 'target']]
        val_p = self.data.loc[self.data.index.isin(test_id), :].copy()[['path', 'target']]
        val_p, _ = train_test_split(val_p, test_size=0.7, random_state=0)  # 30% validation set

        if self.subset_size == 1.0:
            pass
        else:
            train_p, _ = train_test_split(train_p, test_size=1 - self.subset_size, random_state=0)

        dataset_train = Kvasir(self.dataroot, df_data=train_p, transform=self.trans_train)
        dataset_valid = Kvasir(self.dataroot, df_data=val_p, transform=self.trans_valid)

        loader_train = DataLoader(dataset=dataset_train, shuffle=True, batch_size=self.batch_size,
                                  num_workers=self.num_workers)
        loader_valid = DataLoader(dataset=dataset_valid, batch_size=self.batch_size, shuffle=False,
                                  num_workers=self.num_workers)
        return loader_train, loader_valid


if __name__ == '__main__':
    root = '/Users/juanwenwang/Datasets/Kvaisir'
    path = root + '/' + 'metadata.csv'
    batch_size = 2
    subset_size = 1.0
    n_split = 2
    currerent_fold = 0
    num_workers = 0
    noise = 'gaussian'
    mean = 0
    var = 0.01
    amount = 0.05

    dataloader = get_dataloader_kvasir(root, path, batch_size, subset_size, currerent_fold, num_workers, n_split,
                                       noise, mean, var, amount)
    train_loader, val_loader = dataloader.get_data_loader()
    print(iter(val_loader).__next__()[0].shape)
    print(len(train_loader), len(val_loader))