from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
import h5py
from typing import List, Sequence, Optional

class MultiViewArrayDataset(Dataset):
    """Generic multi-view dataset backed by numpy arrays."""

    def __init__(self, views: Sequence[np.ndarray], labels: np.ndarray):
        self.Vs = [np.asarray(v, dtype=np.float32) for v in views]
        self.Y = np.asarray(labels).reshape(-1).astype(np.int64)
        self.num_views = len(self.Vs)
        for i, v in enumerate(self.Vs):
            setattr(self, f"V{i+1}", v)

    def __len__(self):
        return self.Vs[0].shape[0]

    def __getitem__(self, idx):
        xs = [torch.from_numpy(v[idx]) for v in self.Vs]
        return xs, torch.tensor(self.Y[idx], dtype=torch.long), torch.from_numpy(np.array(idx)).long()

    def clone(self):
        return MultiViewArrayDataset([v.copy() for v in self.Vs], self.Y.copy())

    def subset_same_indices(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        return MultiViewArrayDataset([v[indices].copy() for v in self.Vs], self.Y[indices].copy())

    def subset_per_view(self, indices_per_view: Sequence[np.ndarray]):
        views = []
        min_len = min(len(idx) for idx in indices_per_view)
        if min_len == 0:
            raise ValueError("At least one view subset is empty.")
        for v, idx in enumerate(indices_per_view):
            idx = np.asarray(idx, dtype=np.int64)[:min_len]
            views.append(self.Vs[v][idx].copy())
        labels = self.Y[np.asarray(indices_per_view[0], dtype=np.int64)[:min_len]].copy()
        return MultiViewArrayDataset(views, labels)

    def sample_mean(self):
        return [v.mean(axis=0).astype(np.float32) for v in self.Vs]

    def pretrain_sigma(self):
        sigmas = []
        for arr in self.Vs:
            view = arr.astype(np.float32).copy()
            for i in range(view.shape[0]):
                vmax = view[i].max()
                vmin = view[i].min()
                if vmax > vmin:
                    view[i] = (view[i] - vmin) / (vmax - vmin)
                else:
                    view[i] = 0.0
            sigmas.append(view.std(axis=0).astype(np.float32))
        return sigmas


class my(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path)['X1'].transpose().astype(np.float32)
        data2 = scipy.io.loadmat(path)['X2'].transpose().astype(np.float32)
        labels = scipy.io.loadmat(path)['gt']
        self.V1 = data1
        self.V2 = data2
        self.Y = labels

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(
           self.V2[idx])], torch.from_numpy(self.Y[idx]), torch.from_numpy(np.array(idx)).long()

    def percentage_dele(self, road, avail1, avail2, pair):
        if road == 2:
            train_sample1 = []
            train_sample2 = []
            for i in pair:
                train_sample1.append(self.V1[i])
                train_sample2.append(self.V2[i])
            train_sample1 = np.array(train_sample1).astype(np.float32)
            train_sample2 = np.array(train_sample2).astype(np.float32)
            self.V1 = train_sample1
            self.V2 = train_sample2

        elif road == 1:
            avail_sample1 = []
            avail_sample2 = []
            for i in avail1:
                avail_sample1.append(self.V1[i])
            for i in avail2:
                avail_sample2.append(self.V2[i])
            avail_sample1 = np.array(avail_sample1).astype(np.float32)
            avail_sample2 = np.array(avail_sample2).astype(np.float32)
            self.V1 = avail_sample1
            self.V2 = avail_sample2

    def sample_mean(self):
        V1_mean = self.V1.mean(axis=0).astype(np.float32)
        V2_mean = self.V2.mean(axis=0).astype(np.float32)
        return V1_mean, V2_mean

    def pretrain_sigma(self):
        view1 = self.V1.astype(np.float32)
        view2 = self.V2.astype(np.float32)
        for i in range(view1.shape[0]):
            view1[i] = (view1[i] - view1[i].min()) / (view1[i].max() - view1[i].min())
            view2[i] = (view2[i] - view2[i].min()) / (view2[i].max() - view2[i].min())
        sigma1 = view1.std(axis=0)
        sigma2 = view2.std(axis=0)
        return sigma1, sigma2


class myh5py(Dataset):
    def __init__(self, path):
        data1 = np.array(h5py.File(path)['X1']).astype(np.float32)
        data2 = np.array(h5py.File(path)['X2']).astype(np.float32)
        labels = np.array(h5py.File(path)['gt']).transpose().astype(np.float32)
        self.V1 = data1
        self.V2 = data2
        self.Y = labels

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(
           self.V2[idx])], torch.from_numpy(self.Y[idx]), torch.from_numpy(np.array(idx)).long()

    def percentage_dele(self, road, avail1, avail2, pair):
        if road == 2:
            train_sample1 = []
            train_sample2 = []
            for i in pair:
                train_sample1.append(self.V1[i])
                train_sample2.append(self.V2[i])
            train_sample1 = np.array(train_sample1).astype(np.float32)
            train_sample2 = np.array(train_sample2).astype(np.float32)
            self.V1 = train_sample1
            self.V2 = train_sample2

        elif road == 1:
            avail_sample1 = []
            avail_sample2 = []
            for i in avail1:
                avail_sample1.append(self.V1[i])
            for i in avail2:
                avail_sample2.append(self.V2[i])
            avail_sample1 = np.array(avail_sample1).astype(np.float32)
            avail_sample2 = np.array(avail_sample2).astype(np.float32)
            self.V1 = avail_sample1
            self.V2 = avail_sample2

    def sample_mean(self):

        V1_mean = self.V1.mean(axis=0).astype(np.float32)
        V2_mean = self.V2.mean(axis=0).astype(np.float32)
        return V1_mean, V2_mean

    def pretrain_sigma(self):
        view1 = self.V1.astype(np.float32)
        view2 = self.V2.astype(np.float32)
        for i in range(view1.shape[0]):
            view1[i] = (view1[i] - view1[i].min()) / (view1[i].max() - view1[i].min())
            view2[i] = (view2[i] - view2[i].min()) / (view2[i].max() - view2[i].min())
        sigma1 = view1.std(axis=0)
        sigma2 = view2.std(axis=0)
        return sigma1, sigma2


class BDGP(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose()
        self.V1 = data1
        self.V2 = data2
        self.Y = labels

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(
           self.V2[idx])], torch.from_numpy(self.Y[idx]), torch.from_numpy(np.array(idx)).long()

    def percentage_dele(self, road, avail1, avail2, pair):
        if road == 2:
            train_sample1 = []
            train_sample2 = []
            for i in pair:
                train_sample1.append(self.V1[i])
                train_sample2.append(self.V2[i])
            train_sample1 = np.array(train_sample1).astype(np.float32)
            train_sample2 = np.array(train_sample2).astype(np.float32)
            self.V1 = train_sample1
            self.V2 = train_sample2

        elif road == 1:
            avail_sample1 = []
            avail_sample2 = []
            for i in avail1:
                avail_sample1.append(self.V1[i])
            for i in avail2:
                avail_sample2.append(self.V2[i])
            avail_sample1 = np.array(avail_sample1).astype(np.float32)
            avail_sample2 = np.array(avail_sample2).astype(np.float32)
            self.V1 = avail_sample1
            self.V2 = avail_sample2

    def sample_mean(self):
        V1_mean = self.V1.mean(axis=0).astype(np.float32)
        V2_mean = self.V2.mean(axis=0).astype(np.float32)
        return V1_mean, V2_mean

    def pretrain_sigma(self):
        view1 = self.V1.astype(np.float32)
        view2 = self.V2.astype(np.float32)
        for i in range(view1.shape[0]):
            view1[i] = (view1[i] - view1[i].min()) / (view1[i].max() - view1[i].min())
            view2[i] = (view2[i] - view2[i].min()) / (view2[i].max() - view2[i].min())
        sigma1 = view1.std(axis=0)
        sigma2 = view2.std(axis=0)
        return sigma1, sigma2


class CCV(Dataset):
    def __init__(self, path):
        self.data1 = np.load(path+'STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        self.data1 = scaler.fit_transform(self.data1)
        self.data2 = np.load(path+'SIFT.npy').astype(np.float32)
        self.data3 = np.load(path+'MFCC.npy').astype(np.float32)
        self.labels = np.load(path+'label.npy')

    def __len__(self):
        return 6773

    def __getitem__(self, idx):
        x1 = self.data1[idx]
        x2 = self.data2[idx]
        x3 = self.data3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(
           x2), torch.from_numpy(x3)], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class MNIST_USPS(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)            
        self.V2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

    def percentage_dele(self, road, avail1, avail2, pair):
        if road == 2:
            train_sample1 = []
            train_sample2 = []
            for i in pair:
                train_sample1.append(self.V1[i])
                train_sample2.append(self.V2[i])
            train_sample1 = np.array(train_sample1).astype(np.float32)
            train_sample2 = np.array(train_sample2).astype(np.float32)
            self.V1 = train_sample1
            self.V2 = train_sample2

        elif road == 1:
            avail_sample1 = []
            avail_sample2 = []
            for i in avail1:
                avail_sample1.append(self.V1[i])
            for i in avail2:
                avail_sample2.append(self.V2[i])
            avail_sample1 = np.array(avail_sample1).astype(np.float32)
            avail_sample2 = np.array(avail_sample2).astype(np.float32)
            self.V1 = avail_sample1
            self.V2 = avail_sample2

    def sample_mean(self):
        V1_mean = self.V1.mean(axis=0).astype(np.float32)
        V2_mean = self.V2.mean(axis=0).astype(np.float32)
        return V1_mean, V2_mean

    def pretrain_sigma(self):
        view1 = self.V1.astype(np.float32)
        view2 = self.V2.astype(np.float32)
        for i in range(view1.shape[0]):
            view1[i] = (view1[i] - view1[i].min()) / (view1[i].max() - view1[i].min())
            view2[i] = (view2[i] - view2[i].min()) / (view2[i].max() - view2[i].min())
        sigma1 = view1.std(axis=0)
        sigma2 = view2.std(axis=0)
        return sigma1, sigma2


class Fashion(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Fashion.mat')['Y'].astype(np.int32).reshape(10000,)
        self.V1 = scipy.io.loadmat(path + 'Fashion.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Fashion.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Fashion.mat')['X3'].astype(np.float32)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        x3 = self.V3[idx].reshape(784)

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()

class COIL20(MultiViewArrayDataset):
    """
    COIL20.mat structure:
      X: 1x3 cell array
         X[0]: (1440, 32, 32)
         X[1]: (1440, 32, 32)
         X[2]: (1440, 324)
      Y: (1, 1440)
    """

    def __init__(self, path):
        data = scipy.io.loadmat(path)
        X = data['X']
        Y = data['Y'].reshape(-1).astype(np.int64)

        views: List[np.ndarray] = []
        for i in range(X.shape[1]):
            arr = np.array(X[0, i])
            if arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            arr = arr.astype(np.float32)
            if i < 2:
                arr = arr / 255.0
            else:
                scaler = MinMaxScaler()
                arr = scaler.fit_transform(arr).astype(np.float32)
            views.append(arr)

        super().__init__(views, Y)

def load_data(dataset, data_path: Optional[str] = None):
    if dataset == "BDGP":
        dataset = BDGP('./data/')
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "MNIST_USPS":
        dataset = MNIST_USPS('./data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000
    elif dataset == "my_UCI":
        dataset = my(('./data/' + dataset + '.mat'))
        dims = [240, 76]
        view = 2
        class_num = 10
        data_size = 2000
    elif dataset == "COIL20":
        if data_path is None:
            data_path = './COIL20.mat'
        dataset = COIL20(data_path)
        dims = [1024, 1024, 324]
        view = 3
        class_num = 20
        data_size = 1440

    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
