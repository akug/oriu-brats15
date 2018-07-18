#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import numpy as np
import h5py
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from skimage import img_as_float
from skimage.transform import resize

from scipy.ndimage import (gaussian_filter, gaussian_laplace, gaussian_gradient_magnitude)
from skimage.feature import (hessian_matrix, hessian_matrix_eigvals,
                             structure_tensor, structure_tensor_eigvals)
from skimage.exposure import equalize_hist
from skimage.morphology import disk
from skimage.filters.rank import equalize
try:
    from tqdm import tqdm, trange
    print_fn = tqdm.write
except ImportError:
    def tqdm(x):
        return x
    trange = range
    print_fn = print

_save_name = 'brats2015_{}_{}_r{}_std{}.h5'
_save_dir = '/home/alex/Desktop/'
_load_path = '/home/alex/Nextcloud/BraTS/BraTS2015'
_load_path_np = '/home/alex/Nextcloud/BraTS/BraTS2015/numpy/'
_shape = (240, 240)
_scans = ['MR_T1', 'MR_T1c', 'MR_T2', 'MR_Flair', 'all']
_train_split = 0.8
# stats are for the integer images
_stats = {
'MR_Flair': {'sum': 125242999073.0, 'sq_sum': 8.107491724760754e+19, 'mean': 51.19749523887777, 
             'std': 182050.07729015464, 'var': 33142230641.351284, 'max': 5239.0, 'min': 0.0, 'N': 274.0},
'MR_T1': {'sum': 180813802881.0, 'sq_sum': 1.485515459109165e+20, 'mean': 73.91402218600385,
          'std': 246425.81189688717, 'var': 60725680769.04002, 'max': 7779.0, 'min': -551.0, 'N': 274.0},
'MR_T1c': {'sum': 198700755282.0, 'sq_sum': 1.809067540980046e+20, 'mean': 81.22594514510243,
           'std': 271941.19341910875, 'var': 73952012678.2091, 'max': 11737.0, 'min': 0.0, 'N': 274.0},
'MR_T2': {'sum': 230799513203.0, 'sq_sum': 2.414008769547383e+20, 'mean': 94.34744509318669,
          'std': 314135.50851289934, 'var': 98681117708.65787, 'max': 15281.0, 'min': 0.0, 'N': 274.0},
'all': {'sum': 735557070439.0, 'sq_sum': 6.519340942112674e+20, 'mean': 75.17122691579269,
        'std': 258118.69497285827, 'var': 66625260694.491455, 'max': 15281.0, 'min': -551.0, 'N': 1096.0}
}

class Brats15Dataset(Dataset):
    """ Dataset class to load the 3D brain images of the BraTS2015 challenge (original data)
    """
    train_path = "BRATS2015_Training"
    challenge_path = "Testing"

    def __init__(self, path, train, train_split=0.8, which_gg='both', which_scan='all', random_state=-1,
                 transform=transforms.Compose([torch.from_numpy]), preload_data=False, correct=False):
        """

        Args:
            path (str): Should point to a folder named BraTS2015_Training
            train (bool): If True, load training data, if False load testing data
            which_gg (str): Can be 'HGG', 'LGG' or 'both'
            which_scan (str): Can be 'MR_T1', 'MR_T1c', 'MR_T2', 'MR_Flair' or 'all'
            random_state (int): Seed to shuffle the sets before partitioning in train and test set.
                                If no shuffling is desired, pass random_state=-1
            transform (torch.transforms): Transformation done on the input images (not on the labels)
            preload_data (bool): If data should be loaded in memory
        """

        assert which_gg in ['HGG', 'LGG', 'both']
        assert which_scan in _scans
        self.train = train
        self.path = os.path.join(path, self.train_path)
        self.which_gg = which_gg
        self.which_scan = which_scan
        self.train_split = train_split
        self.transform = transform
        self.preload_data = preload_data
        self.correct = correct
        self._paths = []
        self.train_paths = []
        self.test_paths = []
        self.X = None
        self.Y = None

        if not random_state == -1:
            rng = np.random.RandomState(random_state)

        if which_gg == 'both':
            gg = ['HGG', 'LGG']
        else:
            gg = [which_gg]
        for gg_type in gg:
            path = os.path.join(self.path, gg_type)
            for pp in os.listdir(path):
                pp_path = os.path.join(path, pp)
                img_paths = []
                vsd_ids = []
                scan_types = []
                label = ""
                for img_folder in os.listdir(pp_path):
                    img_folder_path = os.path.join(pp_path, img_folder)
                    scan_type = img_folder_path.rsplit('.')[-2]
                    img_path = [os.path.join(img_folder_path, ff)
                                for ff in os.listdir(img_folder_path) if ff.endswith(".mha")][0]
                    if scan_type == "OT":
                        label = img_path
                    else:
                        if (scan_type == self.which_scan or self.which_scan == 'all'):
                            scan_types.append(scan_type)
                            img_paths.append(img_path)
                            vsd_ids.append(img_folder_path.rsplit('.')[-1])
                    ordered = self._order_paths(img_paths, label, vsd_ids, scan_types)
                self._paths += ordered
        if random_state == -1:
            self.train_paths = self._paths[:int(self.train_split*len(self._paths))]
            self.test_paths = self._paths[int(self.train_split*len(self._paths)):]
        else:
            nn = len(self._paths)
            idxs = rng.permutation(nn)
            self.train_paths = [self._paths[idx] for idx in idxs[:int(self.train_split*nn)]]
            self.test_paths = [self._paths[idx] for idx in idxs[int(self.train_split*nn):]]
        if self.train:
            self.paths = self.train_paths
        else:
            self.paths = self.test_paths
        if self.preload_data:
            print("WARN: This needs A LOT of memory (~50 GB)")
            self._preload_data()

    @staticmethod
    def _order_paths(img_paths, label, vsd_ids, scan_types):
        ordered = []
        for img_path, vsd, scan_type in zip(img_paths, vsd_ids, scan_types):
            ordered.append([img_path, label, vsd_ids, scan_type])
        return ordered

    def __getitem__(self, idx):
        if self.preload_data:
            x = self.X[idx]
            y = self.Y[idx]
        else:
            img_path = self.paths[idx][0]
            img_path_label = self.paths[idx][1]
            # need images with shape (Channel, Depth, Height, Width)
            # https://pytorch.org/docs/stable/nn.html#torch.nn.Conv3d
            x = load_and_convert(img_path, correct=self.correct)[None, :]
            if self.correct:
                x = convert_to_float(x, self.which_scan)
            y = load_and_convert(img_path_label, correct=False)[None, :]
            if self.transform is not None:
                x = self.transform(x)
            #y_lab = (y.sum() > 0).astype(np.int32)
            #y = torch.Tensor([y_lab])
            #y = self.transform(y).type(torch.LongTensor)
        return x, y

    def __len__(self):
        return len(self.paths)

    def _preload_data(self):
        self.X = self.transform(np.array([load_and_convert(img_path[0], correct=self.correct)[None, :] for img_path in self.paths]))
        self.Y = self.transform(np.array([load_and_convert(img_path[1], correct=False)[None, :] for img_path in self.paths]))

class Brats15NumpyDataset(Dataset):
    """ Dataset class to load the 2D brain images of the BraTS2015 challenge saved as numpy.ndarray in h5
    """

    def __init__(self, path, train, train_split=0.8, random_state=-1,
                 transform=transforms.Compose([torch.from_numpy]), np_transform=None,
                 np_transform_params=None, preload_data=False, tensor_conversion=True):
        """

        Args:
            path (str): Should point to a folder named BraTS2015_Training
            train (bool): If True, load training data, if False load test data
            random_state (int): Seed to shuffle the sets before partitioning in train and test set.
                                If no shuffling is desired, pass random_state=-1
            transform (torch.transforms): Transformation done on the input images (not on the labels)
            preload_data (bool): If data should be loaded in memory
        """

        self.train = train
        self.path = path
        self.train_split = train_split
        self.transform = transform
        self.np_transform = np_transform
        self.np_transform_params = np_transform_params
        self._np_transform = [] if self.np_transform is None else self.np_transform
        self._np_params = [] if self.np_transform is None else self.np_transform_params
        self.preload_data = preload_data
        self.tensor_conversion = tensor_conversion
        self.X = None
        self.Y = None

        with h5py.File(self.path, 'r') as ff:
            self.len_tot = len(ff) // 2
        _len = int(train_split*(self.len_tot))
        self.idxs = list(range(self.len_tot))
        if random_state > -1:
            rng = np.random.RandomState(random_state)
            self.idxs = rng.permutation(self.idxs)
        if self.train:
            self.idxs = self.idxs[:_len]
        else:
            self.idxs = self.idxs[_len:]
        self.len = len(self.idxs)

    def __getitem__(self, idx):
        if self.preload_data:
            x = self.X[idx]
            y = self.Y[idx]
        else:
            with h5py.File(self.path, 'r') as ff:
                x = np.array(ff["{:0>6}".format(self.idxs[idx])])
                y = np.array(ff["{:0>6}_label".format(self.idxs[idx])])
            for trans, params in zip(self._np_transform, self._np_params):
                x = trans(x, **params)
                y = trans(y, **params)
            # need images with shape (Channel, Height, Width)
            # https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d
            x = x[None, :]
            y = y[None, :]
            if self.tensor_conversion:
                x = self.transform(x).float()
                y = self.transform(y).long()
        return x, y

    def __len__(self):
        return self.len

    def _preload_data(self):
        with h5py.File(self.path, 'r') as ff:
            self.len_tot = len(ff) // 2
            self.X = self.transform(np.array([np.array(ff["{:0>6}".format(self.idxs[idx])])[None, :] for idx in self.idxs]))
            self.Y = self.transform(np.array([np.array(ff["{:0>6}_label".format(self.idxs[idx])])[None, :] for idx in self.idxs]))

class Loader:
    """ Generic loader to load items from a dataset
    """
    def __init__(self, dset, batch_size=1, random_state=-1):
        self.dset = dset
        self.batch_size = batch_size
        self.random_state = random_state
        self._cur = 0
        self._stop = False
        self._len = self.dset.len // batch_size
        self.idxs = np.arange(self.dset.len)
        if random_state > -1:
            rng = np.random.RandomState(random_state)
            self.idxs = rng.permutation(self.idxs)

    def __len__(self):
        return self._len

    def __iter__(self):
        return self

    def __next__(self):
        if self._cur == self.dset.len:
            raise StopIteration
        batches_x = []
        batches_y = []
        for _ in range(self.batch_size):
            try:
                batches_x.append(self.dset[self.idxs[self._cur]][0])
                batches_y.append(self.dset[self.idxs[self._cur]][1])
            except IndexError:
                break
            self._cur += 1
        return np.array(batches_x), np.array(batches_y)

def export_two(load_path, save_path, idxs, train):
    if train:
        #np_transform = [crop, resize]
        #np_params = [{'size': np.multiply(_shape, 0.8).astype(int)},
        #             {'output_shape': np.multiply(_shape, 0.4).astype(int),
        #              'mode': 'constant'}]
        np_transform = None
        np_params = None
    else:
        np_transform = None
        np_params = None
    dset = Brats15NumpyDataset(load_path, train, train_split=0.8, random_state=-1,
                                     transform=None, np_transform=np_transform,
                                     np_transform_params=np_params, tensor_conversion=False)
    with h5py.File(save_path, 'w') as f:
        for ii, idx in enumerate(idxs):
            x, y = dset[idx]
            dset_h5 = f.create_dataset("{:0>6}".format(ii), data=x,
                                       dtype=np.float32)
            dset_h5_lab = f.create_dataset("{:0>6}_label".format(ii), data=y,
                                           dtype=np.int16)

def export_two_train():
    load_path = '/home/alex/Desktop/brats2015_MR_T2_select2_correct.h5'
    save_path = '/home/alex/Desktop/brats2015_MR_T2_two_train.h5'
    idxs = [1137, 2685]
    export_two(load_path, save_path, idxs, True)

def export_two_test():
    load_path = '/home/alex/Desktop/brats2015_MR_T2_select2_correct.h5'
    save_path = '/home/alex/Desktop/brats2015_MR_T2_two_test.h5'
    idxs = [2377, 1634]
    export_two(load_path, save_path, idxs, False)

def crop(x, size):
    ends = np.subtract(x.shape, size).astype(int) // 2
    x_ = x[ends[0]:-ends[0], ends[1]:-ends[1]]
    return x_

def apply_filters(images, sigmas):
    """ Apply multiple filters to 'images'
    """
    filtered_images = []
    for img in images:
        filtered = []
        for sigma in sigmas:
            for conv_filter in [gaussian_filter, gaussian_laplace, gaussian_gradient_magnitude]:
                filtered.append(conv_filter(img, sigma))
            # *_eigenvals has changed from version 0.13 to 0.14.
            try:
                # v. 0.14
                eigs_struc = structure_tensor_eigvals(*structure_tensor(img, sigma=sigma))
                eigs_hess = hessian_matrix_eigvals(hessian_matrix(img, sigma=sigma, order="xy"))
            except TypeError as e:
                # v. 0.13
                eigs_struc = structure_tensor_eigvals(*structure_tensor(img, sigma=sigma))
                eigs_hess = hessian_matrix_eigvals(*hessian_matrix(img, sigma=sigma, order="xy"))
            for eig_h, eig_s in zip(eigs_struc, eigs_hess):
                filtered.append(eig_h)
                filtered.append(eig_s)

        filtered.append(equalize_hist(img))
        #selem = disk(30)
        #filtered.append(equalize(img, selem=selem))
        filtered_images.append(filtered)

    return np.array(filtered_images)

def load_stats(path, which_scan):
    with pd.HDFStore('./brats2015_stats.h5') as store:
        df = store[which_scan]
    return df

def convert_to_float(img, which_scan):
    img = img.astype(float) / _stats[which_scan]['max']
    return img

def load_and_convert(path, correct=False):
    """ Load an image from the BraTS2015 challenge and convert it to numpy
    """
    import SimpleITK as sitk
    #if correct:
    #    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    #    smoother = sitk.GradientAnisotropicDiffusionImageFilter()
    image = sitk.ReadImage(path)
    max_index = image.GetDepth()
    imgs = []
    for i in range(max_index):
        img_s = image[:,:,i]
        if correct:
            #maskImage = sitk.OtsuThreshold(img_s, 0, 1, 200)
            #img_s = sitk.Cast(img_s, sitk.sitkFloat32)
            image_trans = img_s #corrector.Execute(img_s)
            image_trans = img_s #smoother.Execute(image_trans)
            image_trans = img_s
        else:
            image_trans = img_s
        img = sitk.GetArrayFromImage(image_trans)
        imgs.append(img)
    return np.array(imgs, dtype=np.int16)

def tumor_background_ratio(img):
    ratio = (img > 0).sum() / (1.*img.size)
    return ratio

def load_and_save_2d(load_path, save_path, which_gg='both', which_scan='all', random_state=-1, keep_ratio=None, correct=False):
    """ Load the dataset and save the 3D images as 2D images in an h5 file
    """
    dset = Brats15Dataset(load_path, True, 1., which_gg, which_scan, random_state,
                          transform=None, preload_data=False)
    if os.path.isfile(save_path):
        os.remove(save_path)
    with h5py.File(save_path, 'a') as f:
        f.attrs['GG'] = which_gg
        f.attrs['scan'] = which_scan
        if keep_ratio is not None:
            f.attrs['keep_ratio'] = keep_ratio
        counter = 0
        for kk, (data_path, label_path, vsd, scan_type) in enumerate(tqdm(dset.train_paths, 'paths')):
            # Saving as numpy.ndarray instead of torch.Tensor has two advantages:
            # transformations can be done easier (with numpy/skimage)
            # File size is about 40% smaller with numpy array (by test)
            img3d_x = load_and_convert(data_path, correct=correct)
            if correct:
                img3d_x = convert_to_float(img3d_x, which_scan)
            img3d_y = load_and_convert(label_path, correct=False)
            for (img2d_x, img2d_y) in zip(img3d_x, img3d_y):
                if keep_ratio is not None:
                    rel_labels = tumor_background_ratio(img2d_y)
                    if rel_labels < keep_ratio:
                        continue
                #print(img2d_x.dtype, img2d_x.max(), img2d_x.min())
                #img2d_x = img_as_float(img2d_x)
                #print(img2d_x.dtype, img2d_x.max(), img2d_x.min())
                dset_h5 = f.create_dataset("{:0>6}".format(counter), data=img2d_x,
                                           dtype=np.float32, compression="gzip", compression_opts=9)
                dset_h5_lab = f.create_dataset("{:0>6}_label".format(counter), data=img2d_y,
                                               dtype=np.int16, compression="gzip", compression_opts=9)
                dset_h5.attrs['vsd'] = vsd[0]
                dset_h5_lab.attrs['vsd'] = vsd[0]
                dset_h5.attrs['scan'] = scan_type
                dset_h5_lab.attrs['vsd'] = scan_type
                counter += 1

def load_and_save_2d_whole(load_path, save_path, which_gg='both', which_scan='all', random_state=-1, keep_ratio=None, correct=False):
    """ Load the dataset and save the 3D images as 2D images in an h5 file
    """
    from medpy.filter.IntensityRangeStandardization import IntensityRangeStandardization

    dset = Brats15Dataset(load_path, True, 1., which_gg, which_scan, random_state,
                          transform=None, preload_data=False)
    trans = IntensityRangeStandardization(landmarkp=[10, 80], stdrange=[0., 1.])
    imgs = []
    labels = []
    for kk, (data_path, label_path, vsd, scan_type) in enumerate(tqdm(dset.train_paths, 'paths')):
        # Saving as numpy.ndarray instead of torch.Tensor has two advantages:
        # transformations can be done easier (with numpy/skimage)
        # File size is about 40% smaller with numpy array (by test)
        img3d_x = load_and_convert(data_path, correct=correct)
        if correct:
            img3d_x = convert_to_float(img3d_x, which_scan)
        img3d_y = load_and_convert(label_path, correct=False)
        for (img2d_x, img2d_y) in zip(img3d_x, img3d_y):
            if keep_ratio is not None:
                rel_labels = tumor_background_ratio(img2d_y)
                if rel_labels < keep_ratio:
                    continue
            imgs.append(img2d_x)
            labels.append(img2d_y)
    imgs_trans = imgs #trans.train_transform(imgs)
    if os.path.isfile(save_path):
        os.remove(save_path)
    with h5py.File(save_path, 'a') as f:
        f.attrs['GG'] = which_gg
        f.attrs['scan'] = which_scan
        if keep_ratio is not None:
            f.attrs['keep_ratio'] = keep_ratio
        for ii, (img2d_x, img2d_y) in enumerate(zip(imgs_trans, labels)):
            dset_h5 = f.create_dataset("{:0>6}".format(ii), data=img2d_x,
                                       dtype=np.float32, compression="gzip", compression_opts=9)
            dset_h5_lab = f.create_dataset("{:0>6}_label".format(ii), data=img2d_y,
                                           dtype=np.int16, compression="gzip", compression_opts=9)

def save_stats(load_path, which_scan, save_path):
    stats = {'x': {'sum': 0., 'sq_sum': 0., 'mean': 0., 'std': 0.,
                   'var': 0., 'max': -np.inf, 'min': np.inf, 'N': 0}}
    dset = Brats15Dataset(load_path, True, 1., 'both', which_scan, -1, transform=None, preload_data=False)
    size = dset[0][0].size
    for x, _ in tqdm(dset, desc='img'):
        stats['x']['N'] += 1
        ss = x.sum()
        stats['x']['sum'] += ss
        stats['x']['sq_sum'] += ss**2
        stats['x']['max'] = np.max([np.max(x), stats['x']['max']])
        stats['x']['min'] = np.min([np.min(x), stats['x']['min']])
    stats['x']['mean'] = 1.*stats['x']['sum'] / (stats['x']['N']*size)
    stats['x']['var'] = 1.*stats['x']['sq_sum'] / (stats['x']['N']*size) - (1.*stats['x']['sum'] / (stats['x']['N']*size))**2
    stats['x']['std'] = np.sqrt(stats['x']['var'])
    df = pd.DataFrame.from_dict(stats, orient='index')
    df.to_hdf(save_path, which_scan, mode='a')

def collect_stats(load_path, save_path):
    for which_scan in tqdm(_scans, 'scan'):
        save_stats(load_path, which_scan, save_path)

def convert_and_save_2d(load_path, save_path, np_transform, np_transform_params, sigmas, n_rand_pixels=None):
    """ Load a numpy dataset, apply the filters and save the results in a new dataset
    """
    dset = Brats15NumpyDataset(load_path, True, train_split=_train_split, random_state=-1,
                 transform=None, np_transform=np_transform,
                 np_transform_params=np_transform_params, tensor_conversion=False)
    if n_rand_pixels is None:
        tmpfile = '/tmp/test/temp.h5'
        if os.path.isfile(tmpfile):
            os.remove(tmpfile)
        with h5py.File(tmpfile, 'a') as f:
            for ii, (img, label) in enumerate(tqdm(dset, desc='img')):
                img = apply_filters([np.squeeze(img, 0)], sigmas)
                n_feat = img.shape[1]
                img = img.swapaxes(0,1).reshape(n_feat, -1)
                label = label.ravel()
                dset_h5 = f.create_dataset("{:0>6}".format(ii), data=img,
                                           dtype=np.float32, compression="lzf")
                dset_h5_lab = f.create_dataset("{:0>6}_label".format(ii), data=label,
                                               dtype=np.int16, compression="lzf")
        with h5py.File(tmpfile, 'r') as f:
            imgs = np.concatenate([f[key] for key in f if not ("label" in key)], 1)
            labels = np.concatenate([np.array(f[key]) for key in f if "label" in key])
        if os.path.isfile(save_path):
            os.remove(save_path)
        with h5py.File(save_path, 'a') as f:
            #dset_h5 = f.create_dataset("imgs", data=imgs,
            #                           dtype=np.float32, compression="gzip", compression_opts=9)
            #dset_h5_lab = f.create_dataset("labels", data=labels,
            #                               dtype=np.int16, compression="gzip", compression_opts=9)
            dset_h5 = f.create_dataset("imgs", data=imgs,
                                       dtype=np.float32, compression="lzf")
            dset_h5_lab = f.create_dataset("labels", data=labels,
                                           dtype=np.int16, compression="lzf")
    else:
        rng = np.random.RandomState(4125215)
        features = []
        labels = []
        count = 0
        pix_per_round = 50
        while count < n_rand_pixels:
            for ii, (img, label) in enumerate(tqdm(dset, desc='img')):
                feature = apply_filters([np.squeeze(img, 0)], sigmas)
                n_feat = feature.shape[1]
                feature = feature.swapaxes(0,1).reshape(n_feat, -1).T
                label = label.ravel()
                mask_t = label>0
                mask_b = label==0
                idxs = np.arange(len(label))
                idxs_t = rng.permutation(idxs[mask_t])[:pix_per_round]
                labels.append(label[idxs_t])
                features.append(feature[idxs_t])
                idxs_b = rng.permutation(idxs[mask_b])[:pix_per_round]
                labels.append(label[idxs_b])
                features.append(feature[idxs_b])
                count += 2*pix_per_round
            print_fn(str(count))
        labels = np.concatenate(labels, 0)
        features = np.concatenate(features, 0)
        if len(labels) > n_rand_pixels:
            labels = labels[:n_rand_pixels]
            features = features[:n_rand_pixels]
        with h5py.File(save_path, 'w') as f:
            dset_h5 = f.create_dataset("imgs", data=features,
                                       dtype=np.float32, compression="lzf")
            dset_h5_lab = f.create_dataset("labels", data=labels,
                                           dtype=np.int16, compression="lzf")

def test_numpy_dset():
    """ Some checks to see that the creation of the h5 file worked
    """
    which_scan = 'all'
    save_path = os.path.join(_save_dir, _save_name.format(which_scan))
    with h5py.File('./brats2015_all.h5', 'r') as ff:
        print(len(ff))
        print([[kk, f.attrs[kk]] for kk in f.attrs])
        print(list(ff.keys())[-1])
        d = ff[list(ff.keys())[4]]
        print(d.shape)
        print([[kk, d.attrs[kk]] for kk in d.attrs])
        # this takes some time, approx. 2 min
        keys = [f[f_key].attrs.get('scan', '') for f_key in f]
        print(np.unique(keys))

def prepare_2d(save_dir, which_gg='LGG', keep_ratio=None, correct=False, int_stand=False):
    random_state = -1
    for which_scan in tqdm(_scans, 'scan'):
        save_path = os.path.join(save_dir, _save_name.format(which_scan, which_gg, int(1000*keep_ratio), int(int_stand)))
        if int_stand:
            load_and_save_2d_whole(_load_path, save_path, which_gg, which_scan, random_state, keep_ratio, correct)
        else:
            load_and_save_2d(_load_path, save_path, which_gg, which_scan, random_state, keep_ratio, correct)

def convert_2d(load_dir, save_dir, n_rand_pixels=None):
    sigmas = [0.3, 0.7, 1., 3.5]
    np_transform = [crop]
    np_transform_params = [{'size': np.multiply(_shape, 0.8).astype(int)}]
    _scans = ['MR_Flair'] #['MR_T1', 'MR_T1c', 'MR_T2', 'MR_Flair', 'all']
    for which_scan in tqdm(_scans, 'scan'):
        #load_path = os.path.join(load_dir, _save_name.format(which_scan))
        load_path = load_dir #os.path.join(load_dir, _save_name.format(which_scan))
        save_path = os.path.join(save_dir, "brats15_{}_LGG_np.h5".format(which_scan))
        convert_and_save_2d(load_path, save_path, np_transform, np_transform_params, sigmas, n_rand_pixels)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", "-p", action='store_true',
                        help="prepare the raw BraTS2015 data by converting it into h5 format")
    parser.add_argument("--keep", type=float, default=None,
                        help="Keep only images with a tumor-background ratio bigger than 'keep'")
    parser.add_argument("--correct", action='store_true',
                        help="Correct the images using N4 bias field correction and Gradient anisotropic diffusion")
    parser.add_argument("--convert", "-c", action='store_true',
                        help="Convert an h5 dataset with images and labels to an h5 dataset with "
                             "transformed and filtered images")
    parser.add_argument("--collect", action='store_true',
                        help="Collect mean, std etc. from the datasets")
    parser.add_argument("--export_two", default=False,
                        help="Export two images")
    parser.add_argument("--int_stand", action='store_true',
                        help="load all images and do Intensity Range Standarization before saving")
    parser.add_argument("--n_rand_pix", type=int, default=None,
                        help="When converting, only choose n_rand_pix")
    parser.add_argument("--load_path", type=str, default=_load_path_np,
                        help="Path to load data from for converting")
    args  = parser.parse_args()
    return args

def main():
    args = parse_args()
    prepare = args.prepare
    convert = args.convert
    keep_ratio = args.keep
    correct = args.correct
    collect = args.collect
    if prepare:
        prepare_2d(_save_dir, keep_ratio=keep_ratio, correct=correct, int_stand=args.int_stand)
    if convert:
        convert_2d(args.load_path, _save_dir, args.n_rand_pix)
    if collect:
        save_path = os.path.join(_save_dir, "brats2015_stats.h5")
        collect_stats(_load_path, save_path)
    if args.export_two == 'train':
        export_two_train()
    elif args.export_two == 'test':
        export_two_test()

if __name__ == '__main__':
    main()
