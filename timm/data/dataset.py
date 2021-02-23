from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data

import os
import re
import torch
import tarfile
import pandas as pd
import numpy as np
from copy import deepcopy
from collections import defaultdict

from PIL import Image


IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def find_images_and_targets(folder, types=IMG_EXTENSIONS, class_to_idx=None, leaf_name_only=True, sort=True):
    labels = []
    filenames = []
    for root, subdirs, files in os.walk(folder, topdown=False):
        rel_path = os.path.relpath(root, folder) if (root != folder) else ''
        label = os.path.basename(rel_path) if leaf_name_only else rel_path.replace(os.path.sep, '_')
        for f in files:
            base, ext = os.path.splitext(f)
            if ext.lower() in types:
                filenames.append(os.path.join(root, f))
                labels.append(label)
    if class_to_idx is None:
        # building class index
        unique_labels = set(labels)
        sorted_labels = list(sorted(unique_labels, key=natural_key))
        class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
    images_and_targets = zip(filenames, [class_to_idx[l] for l in labels])
    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    return images_and_targets, class_to_idx


def load_class_map(filename, root=''):
    class_to_idx = {}
    class_map_path = filename
    if not os.path.exists(class_map_path):
        class_map_path = os.path.join(root, filename)
        assert os.path.exists(class_map_path), 'Cannot locate specified class map file (%s)' % filename
    class_map_ext = os.path.splitext(filename)[-1].lower()
    if class_map_ext == '.txt':
        with open(class_map_path) as f:
            class_to_idx = {v.strip(): k for k, v in enumerate(f)}
    else:
        assert False, 'Unsupported class map extension'
    return class_to_idx


class CsvDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, transform=None, single_view=False,
            data_percent=80, reverse_order=False):
        super().__init__()
        self.csv = pd.read_csv(file_path)
        self.basedir = os.path.dirname(file_path)
        kfolds=np.unique(self.csv['kfold'])
        kfolds.sort()
        if reverse_order:
            kfolds = kfolds[::-1]
        assert data_percent<=80, "you should leave 20% for val/test"
        kfolds = kfolds[:int(np.ceil(len(kfolds)*data_percent/100))]
        self.csv=self.csv[np.isin(self.csv['kfold'], kfolds)]
        #remap instance_id
        remap = {id:i for i,id in enumerate(np.unique(self.csv['instance_id']))}
        self.csv['instance_id'] = self.csv['instance_id'].map(remap)

        self.transform = transform
        self.classes = np.unique(self.csv['class'])
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        self._remap_indices()
        self.single_view = single_view

        n_instances = len(np.unique(self.csv['instance']))
        n_images = n_instances if self.single_view else len(self.csv)
        print(f"CSVDatas: using folds {kfolds.min()}:{kfolds.max()}")
        print(f"CSVDatas: {self}")

    def _remap_indices(self):
        remap = {id:i for i,id in enumerate(np.unique(self.csv['instance_id']))}
        self.csv['instance_id'] = self.csv['instance_id'].map(remap)

    def __len__(self):
        return len(np.unique(self.csv['instance']))

    def __getitem__(self, item):
        dat = self.csv[self.csv['instance_id']==item]
        assert len(dat)>0

        ind = 0 if self.single_view else np.random.randint(len(dat))
        dat = dat.iloc[ind]
        path = os.path.join(self.basedir, dat['file_path'])
        try:
            img = Image.open(path).convert('RGB')
        except:
            print(f'Warning : failed to loader {path}')
            img = Image.new('RGB', (100,100))

        if self.transform is not None:
            img = self.transform(img)
        return img, self.class_to_idx[dat['class']]

    def __str__(self):
        n_instances = len(np.unique(self.csv['instance']))
        n_images = n_instances if self.single_view else len(self.csv)
        return (f"{n_instances} instances, {n_images} images")


    def split_dataset(self, split_ratio, ds2_single_view=None, **kwargs):
        dataset1 = deepcopy(self)
        dataset2 = deepcopy(self)
        folds = np.unique(self.csv['kfold'])
        folds.sort()
        n_folds1 = int(len(folds)*split_ratio)
        thresh = folds[n_folds1]
        mask = dataset1.csv['kfold']<thresh
        dataset1.csv = dataset1.csv[mask]
        dataset2.csv = dataset2.csv[~mask]
        dataset1._remap_indices()
        dataset2._remap_indices()
        if ds2_single_view is not None:
            dataset2.single_view = ds2_single_view

        for i, dataset in enumerate([dataset1, dataset2]):
            n_instances = len(np.unique(dataset.csv['instance']))
            n_images = n_instances if dataset.single_view else len(dataset.csv)
            print(f"dataset{i} : {dataset}")

        return dataset1, dataset2

class Dataset(data.Dataset):

    def __init__(
            self,
            root,
            load_bytes=False,
            transform=None,
            class_map=''):

        class_to_idx = None
        if class_map:
            class_to_idx = load_class_map(class_map, root)
        images, class_to_idx = find_images_and_targets(root, class_to_idx=class_to_idx)
        if len(images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.samples = images
        self.imgs = self.samples  # torchvision ImageFolder compat
        self.class_to_idx = class_to_idx
        self.load_bytes = load_bytes
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            img = open(path, 'rb').read() if self.load_bytes else Image.open(path).convert('RGB')
        except:
            print(f'Warning : failed to loader {path}')
            img = Image.new('RGB', (100,100))    
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.zeros(1).long()
        return img, target

    def __len__(self):
        return len(self.imgs)

    def filenames(self, indices=[], basename=False):
        if indices:
            if basename:
                return [os.path.basename(self.samples[i][0]) for i in indices]
            else:
                return [self.samples[i][0] for i in indices]
        else:
            if basename:
                return [os.path.basename(x[0]) for x in self.samples]
            else:
                return [x[0] for x in self.samples]

    def split_dataset(self, split_ratio, **kwargs):
        dataset1 = deepcopy(self)
        dataset2 = deepcopy(self)
        target_dict = defaultdict(list)
        for image, target in self.samples:
            target_dict[target].append(image)
        dataset1.samples = []
        dataset2.samples = []
        for k, v in target_dict.items():
            num_elem1 = int(len(v) * split_ratio)
            num_elem2 = len(v)-num_elem1
            dataset1.samples.extend(zip(v[:num_elem1], [k]*num_elem1))
            dataset2.samples.extend(zip(v[num_elem1:], [k]*num_elem2))
        dataset1.imgs = dataset1.samples
        dataset2.imgs = dataset2.samples
        return dataset1, dataset2

def _extract_tar_info(tarfile, class_to_idx=None, sort=True):
    files = []
    labels = []
    for ti in tarfile.getmembers():
        if not ti.isfile():
            continue
        dirname, basename = os.path.split(ti.path)
        label = os.path.basename(dirname)
        ext = os.path.splitext(basename)[1]
        if ext.lower() in IMG_EXTENSIONS:
            files.append(ti)
            labels.append(label)
    if class_to_idx is None:
        unique_labels = set(labels)
        sorted_labels = list(sorted(unique_labels, key=natural_key))
        class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
    tarinfo_and_targets = zip(files, [class_to_idx[l] for l in labels])
    if sort:
        tarinfo_and_targets = sorted(tarinfo_and_targets, key=lambda k: natural_key(k[0].path))
    return tarinfo_and_targets, class_to_idx


class DatasetTar(data.Dataset):

    def __init__(self, root, load_bytes=False, transform=None, class_map=''):

        class_to_idx = None
        if class_map:
            class_to_idx = load_class_map(class_map, root)
        assert os.path.isfile(root)
        self.root = root
        with tarfile.open(root) as tf:  # cannot keep this open across processes, reopen later
            self.samples, self.class_to_idx = _extract_tar_info(tf, class_to_idx)
        self.tarfile = None  # lazy init in __getitem__
        self.load_bytes = load_bytes
        self.transform = transform

    def __getitem__(self, index):
        if self.tarfile is None:
            self.tarfile = tarfile.open(self.root)
        tarinfo, target = self.samples[index]
        iob = self.tarfile.extractfile(tarinfo)
        img = iob.read() if self.load_bytes else Image.open(iob).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.zeros(1).long()
        return img, target

    def __len__(self):
        return len(self.samples)


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __init__(self, dataset, num_splits=2):
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits

    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)

    def __getitem__(self, i):
        x, y = self.dataset[i]  # all splits share the same dataset base transform
        x_list = [self._normalize(x)]  # first split only normalizes (this is the 'clean' split)
        # run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y

    def __len__(self):
        return len(self.dataset)
