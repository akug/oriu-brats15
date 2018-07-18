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
    import cPickle as pkl
except ImportError:
    import pickle as pkl

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

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

def onehot(y, n_classes):
    return np.eye(n_classes)[y]

def dice(pred, label, n_classes):
    epsilon = 1.
    pred_1h = onehot(pred, n_classes)
    label_1h = onehot(label, n_classes)
    intersect = (pred_1h*label_1h).sum(0)
    area = pred_1h.sum(0) + label_1h.sum(0)
    return (2.*intersect+epsilon) / (area+epsilon)

def prepare(x, y, sigmas):
    x = x.numpy()
    y = y.numpy().ravel().astype(np.int16)
    if sigmas is not None:
        x = np.squeeze(x, 1)
        x_filter = apply_filters(x, sigmas)
        num_filters = x_filter.shape[1]
    else:
        num_filters = 1
    x_filter = x_filter.swapaxes(0, 1).reshape(num_filters, -1).T
    return x_filter, y

def train_forest_sequential(rfc, dset, sigmas, batch_size, random_state=12831, n_estimators=100):
    dloader = DataLoader(dset, batch_size=batch_size, shuffle=True)
    #dloader = dataloader(dset, batch_size=batch_size, random_state=random_state)
    for ii, (x, y) in enumerate(tqdm(dloader, desc='step'), 1):
        rfc.set_params(n_estimators=ii*n_estimators)
        t_now = time.time()
        x_filter, y = prepare(x, y, sigmas)
        if rfc.n_classes_ == 2:
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
    n_classes = rfc.n_classes_
    if n_classes == 2:
        y[y>0] = 1
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
    parser.add_argument("--train_split", type=float, default=_train_split,
                        help="split ratio of train and test set")
    parser.add_argument("--test", default=None,
                        help="When True, load test set and test with classifier given with this argument")
    parser.add_argument("--sample_imgs", action='store_true',
                        help="Save some sample images with their labels when testing")

    args  = parser.parse_args()
    return args

def test(dset, sigmas, load_path=None, rfc=None, sample_imgs=False):
    if load_path is not None:
        rfc = joblib.load(load_path)
    n_classes = rfc.n_classes_
    if sample_imgs:
        rng = np.random.RandomState(425039)
        idxs = rng.randint(len(dset), size=10)
        data = {'imgs': [], 'preds': [], 'trues': []}
    else:
        idxs = [-1]
    dloader = DataLoader(dset, batch_size=1, shuffle=False)
    dice_scores = []
    preds = []
    for ii, (x, y) in enumerate(tqdm(dloader, desc='step')):
        x_filter, y_f = prepare(x, y, sigmas)
        y_pred = rfc.predict(x_filter)
        preds.append(y_pred.reshape(y.shape[2:]))
        if n_classes == 2:
            y_f[y_f>0] = 1
        dice_score = dice(y_pred, y_f, n_classes)
        dice_scores.append(dice_score)
        if sample_imgs and ii in idxs:
            data['imgs'].append(np.squeeze(x.numpy()))
            data['preds'].append(np.squeeze(y_pred.reshape(y.shape[2:])))
            data['trues'].append(np.squeeze(y))
            fig, axes = plt.subplots(1, 3, figsize=(5,5))
            axes = axes.flatten()
            axes[0].imshow(data['imgs'][-1], cmap='gray')
            axes[1].imshow(data['preds'][-1])
            axes[2].imshow(data['trues'][-1])
            axes[0].set_title('Image')
            axes[1].set_title('Prediction')
            axes[2].set_title('True')
            plt.tight_layout()
            fig.savefig('./test_image{}'.format(ii), dpi=200.)
    np.savez_compressed('./test_data.npz', pred=preds, dice=dice_score)
    if sample_imgs:
        with open('./test_imgs.pkl', 'wb') as handle:
            pkl.dump(data, handle)

def main():
    _n_classes = 2
    sigmas = [0.3, 0.7, 1., 3.5]
    args = parse_args()
    path = args.path
    _train_split = args.train_split
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
            #np_transform = [crop, resize]
            #np_params = [{'size': np.multiply(shape, 0.8).astype(int)},
            #             {'output_shape': np.multiply(shape, 0.4).astype(int),
            #              'mode': 'constant'}]
            #np_transform = None
            #np_params = None
            np_transform = [crop]
            np_params = [{'size': np.multiply(shape, 0.8).astype(int)}]

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
        test(dset_test, sigmas, load_path=args.test, sample_imgs=args.sample_imgs)

if __name__ == '__main__':
    main()
