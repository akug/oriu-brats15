#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
_thispath = os.path.dirname(os.path.realpath(__file__))
import h5py
import time
import sys
sys.path.insert(0, _thispath)
import argparse
import numpy as np

try:
    from tqdm import tqdm, trange
    print_fn = tqdm.write
except ImportError:
    def tqdm(x):
        return x
    trange = range
    print_fn = print

from sklearn.ensemble import RandomForestClassifier

from skimage.transform import resize
from sklearn.externals import joblib

from torch.utils.data import DataLoader

from load import Brats15NumpyDataset, crop, apply_filters

_train_split = 0.8
_random_state = -1


def prepare(x, y, sigmas):
    x = x.numpy()
    y = y.numpy().ravel().astype(np.int32)
    x = np.squeeze(x)
    if x.ndim == 2:
        x = x[None, :]
    if sigmas is not None:
        x_filter = apply_filters(x, sigmas)
        num_filters = x_filter.shape[1]
    else:
        x_filter = x[None, :]
        num_filters = 1
    x_filter = x_filter.swapaxes(0, 1).reshape(-1, num_filters)
    return x_filter, y

def train_forest_sequential(rfc, dset, sigmas, batch_size, random_state=12831, n_estimators=100):
    dloader = DataLoader(dset, batch_size=batch_size, shuffle=True)
    #dloader = dataloader(dset, batch_size=batch_size, random_state=random_state)
    for ii, (x, y) in enumerate(tqdm(dloader, desc='step'), 1):
        rfc.set_params(n_estimators=ii*n_estimators)
        t_now = time.time()
        x_filter, y = prepare(x, y, sigmas)
        y[y>0] = 1
        print_fn("Filter {}".format(time.time()-t_now))
        t_now = time.time()
        rfc.fit(x_filter, y)
        print_fn("Fit {}".format(time.time()-t_now))
        t_now = time.time()
    return rfc

def train_forest_whole(rfc, path):
    with h5py.File(path, 'r') as f:
        x = np.array(f['imgs'])
        y = np.array(f['labels'])
    rfc.fit(x, y)
    return rfc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", default='./data',
                        help="Path of the training/test data")
    parser.add_argument("--whole", action='store_true',
                        help="When given, train the Random Forest Classifier "
            "using the whole dataset at once. This requires another dataset where the "
            "filters are already computed.")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="batch size")
    parser.add_argument("--test", default=None,
                        help="When True, load test set and test with classifier given with this argument")
    args  = parser.parse_args()
    return args

def test(dset, n_classes, sigmas, load_path=None, rfc=None):
    if load_path is not None:
        rfc = joblib.load(load_path)
    dloader = DataLoader(dset, batch_size=1, shuffle=False)
    sums = [0., 0.]
    for ii, (x, y) in enumerate(tqdm(dloader, desc='step')):
        x_filter, y = prepare(x, y, sigmas)
        y_pred = rfc.predict(x_filter)
        y_pred_1h = np.eye(n_classes)[y_pred]
        y[y>0] = 1
        y_1h = np.eye(n_classes)[y]
        #np.savez_compressed('./test.npz', x=x.numpy(), y_pred=y_pred, y=y)
        #break
        sums[0] += (y_pred>0).sum()
        sums[1] += (y>0).sum()
    np.savez_compressed('./test_sum.npz', sums=sums)

def main():
    _n_classes = 2
    sigmas = [1.] #[0.7, 1., 1.6, 3.5, 5., 10.]
    args = parse_args()
    path = args.path
    if args.test is None:
        whole = args.whole
        if whole:
            rfc = RandomForestClassifier(n_estimators=200, n_jobs=-1)
            rfc.n_classes_ = _n_classes
            rfc = train_forest_whole(rfc, path)
        else:
            batch_size = args.batch_size
            n_estimators = 100
            shape = Brats15NumpyDataset(path, True, 1., -1)[0][0].shape[1:]
            np_transform = [crop, resize]
            np_params = [{'size': np.multiply(shape, 0.8).astype(int)},
                         {'output_shape': np.multiply(shape, 0.2).astype(int),
                          'mode': 'constant'}]

            dset_train = Brats15NumpyDataset(path, True, _train_split, _random_state, transform=None,
                                             np_transform=np_transform, np_transform_params=np_params,
                                             tensor_conversion=False)

            rfc = RandomForestClassifier(n_estimators=0, warm_start=True, n_jobs=-1)
            rfc.n_classes_ = _n_classes
            rfc = train_forest_sequential(rfc, dset_train, sigmas, batch_size=batch_size, n_estimators=n_estimators)
        joblib.dump(rfc, 'rfc_test_{}.pkl'.format(int(whole)))
    else:
        dset_test = Brats15NumpyDataset(path, False, _train_split, _random_state, transform=None,
                                        np_transform=None, np_transform_params=None,
                                        tensor_conversion=False)
        test(dset_test, _n_classes, sigmas, load_path=args.test)

if __name__ == '__main__':
    main()
