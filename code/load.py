#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import SimpleITK as sitk
import os
import argparse
import numpy as np
import h5py

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from skimage import img_as_float
from skimage.transform import resize

from scipy.ndimage import (gaussian_filter, gaussian_laplace, gaussian_gradient_magnitude)
from skimage.feature import (hessian_matrix, hessian_matrix_eigvals,
                             structure_tensor, structure_tensor_eigvals)
try:
    from tqdm import tqdm, trange
    print_fn = tqdm.write
except ImportError:
    def tqdm(x):
        return x
    trange = range
    print_fn = print

_save_name = 'brats2015_{}_select_correct.h5'
_save_dir = '/home/alex/Desktop/'
_load_path = '/home/alex/Nextcloud/BraTS/BraTS2015'
_load_path_np = '/home/alex/Nextcloud/BraTS/BraTS2015/numpy/'
_shape = (240, 240)
_scans = ['MR_T1', 'MR_T1c', 'MR_T2', 'MR_Flair', 'all']

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
            y = load_and_convert(img_path_label, correct=False)[None, :]
            x = self.transform(x).type(torch.FloatTensor)
            y_lab = (y.sum() > 0).astype(np.int32)
            y = torch.Tensor([y_lab]).type(torch.LongTensor)
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
        filtered_images.append(filtered)
    return np.array(filtered_images)

def load_and_convert(path, correct=False):
    """ Load an image from the BraTS2015 challenge and convert it to numpy
    """
    if correct:
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        smoother = GradientAnisotropicDiffusionImageFilter()
    image = sitk.ReadImage(path)
    max_index = image.GetDepth()
    imgs = []
    for i in range(max_index):
        img_s = image[:,:,i]
        if correct:
            maskImage = sitk.OtsuThreshold(img_s, 0, 1, 200)
            img_s = sitk.Cast(img_s, sitk.sitkFloat32)
            image_trans = corrector.Execute(img_s, maskImage)
            image_trans = smoother.Execute(image_trans)
        else:
            image_trans = img_s
        img = sitk.GetArrayFromImage(image_trans)
        imgs.append(img)
    return np.array(imgs)

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
            img3d_y = load_and_convert(label_path, correct=False)
            for (img2d_x, img2d_y) in zip(img3d_x, img3d_y):
                if keep_ratio is not None:
                    rel_labels = (img2d_y > 0).sum() / (1.*img2d_y.size)
                    if rel_labels < keep_ratio:
                        continue
                img2d_x = img_as_float(img2d_x)
                dset_h5 = f.create_dataset("{:0>6}".format(counter), data=img2d_x,
                                           dtype=np.float32, compression="gzip", compression_opts=9)
                dset_h5_lab = f.create_dataset("{:0>6}_label".format(counter), data=img2d_y,
                                               dtype=np.int16, compression="gzip", compression_opts=9)
                dset_h5.attrs['vsd'] = vsd[0]
                dset_h5_lab.attrs['vsd'] = vsd[0]
                dset_h5.attrs['scan'] = scan_type
                dset_h5_lab.attrs['vsd'] = scan_type
                counter += 1

def convert_and_save_2d(load_path, save_path, np_transform, np_transform_params, sigmas):
    """ Load a numpy dataset, apply the filters and save the results in a new dataset
    """
    dset = Brats15NumpyDataset(load_path, True, train_split=1., random_state=-1,
                 transform=None, np_transform=np_transform,
                 np_transform_params=np_transform_params, tensor_conversion=False)
    tmpfile = '/tmp/test/temp.h5'
    if os.path.isfile(tmpfile):
        os.remove(tmpfile)
    with h5py.File(tmpfile, 'a') as f:
        for ii, (img, label) in enumerate(tqdm(dset, desc='img')):
            img = apply_filters([np.squeeze(img)], sigmas).swapaxes(0,1).reshape(7*len(sigmas), -1)
            label = label.ravel()
            #dset_h5 = f.create_dataset("{:0>6}".format(ii), data=img,
            #                           dtype=np.float32, compression="gzip", compression_opts=9)
            #dset_h5_lab = f.create_dataset("{:0>6}_label".format(ii), data=label,
            #                               dtype=np.int16, compression="gzip", compression_opts=9)
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

def prepare_2d(save_dir, keep_ratio=None, correct=False):
    random_state = -1
    which_gg = 'both'
    for which_scan in tqdm(_scans, 'scan'):
        save_path = os.path.join(save_dir, _save_name.format(which_scan))
        load_and_save_2d(_load_path, save_path, which_gg, which_scan, random_state, keep_ratio, correct)

def convert_2d(load_dir, save_dir):
    sigmas = [1.]
    np_transform = [crop, resize]
    np_transform_params = [{'size': np.multiply(_shape, 0.8).astype(int)},
                        {'output_shape': np.multiply(_shape, 0.4).astype(int),
                         'mode': 'constant'}]
    _scans = ['MR_Flair'] #['MR_T1', 'MR_T1c', 'MR_T2', 'MR_Flair', 'all']
    for which_scan in tqdm(_scans, 'scan'):
        load_path = os.path.join(load_dir, _save_name.format(which_scan))
        save_path = os.path.join(save_dir, "brats15_{}_np.h5".format(which_scan))
        convert_and_save_2d(load_path, save_path, np_transform, np_transform_params, sigmas)

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
    args  = parser.parse_args()
    return args

def main():
    args = parse_args()
    prepare = args.prepare
    convert = args.convert
    keep_ratio = args.keep
    if prepare:
        prepare_2d(_save_dir, keep_ratio)
    if convert:
        convert_2d(_load_path_np, _save_dir)

if __name__ == '__main__':
    main()
