import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transforms import get_transforms
from pathlib import Path
import gc
import pickle
HEIGHT = 137
WIDTH = 236
SIZE=128

def laod_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def prepare_image(data_type='train',
                  submission=False, indices=[0, 1, 2, 3]):
    datadir = Path('../input/bengaliai-cv19')
    featherdir = Path('../input/bengaliaicv19feather')
    outdir = Path('.')
    assert data_type in ['train', 'test']
    if submission:
        image_df_list = [pd.read_parquet(datadir / f'{data_type}_image_data_{i}.parquet')
                         for i in indices]
    else:
        image_df_list = [pd.read_feather(featherdir / f'{data_type}_image_data_{i}.feather')
                         for i in indices]

    # print('image_df_list', len(image_df_list))

    images = [df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH) for df in image_df_list]
    del image_df_list
    gc.collect()
    images = np.concatenate(images, axis=0)
    return images.copy()

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(img0, size=SIZE, pad=16):
    #crop a box around pixels large than the threshold 
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax,xmin:xmax]
    #remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img,(size,size))

def prepare_pkl_images(indices=[0, 1, 2, 3]):
    imgs = []
    PATHS = ['../input/bengaliai-cv19/test_image_data_0.parquet',
            '../input/bengaliai-cv19/test_image_data_1.parquet',
            '../input/bengaliai-cv19/test_image_data_2.parquet',
            '../input/bengaliai-cv19/test_image_data_3.parquet']
    PATHS = PATHS[indices]
    for fname in PATHS:
        df = pd.read_parquet(fname)
        #the input is inverted
        data = 255 - df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)
        for idx in tqdm(range(len(df))):
            name = df.iloc[idx,0]
            #normalize each image by its max val
            img = (data[idx]*(255.0/data[idx].max())).astype(np.uint8)
            img = crop_resize(img)
            imgs.append(img)
    return np.stack(imgs)

class TrainDataset(Dataset):
    def __init__(self, df, centercrop, transforms, unique_label, unique_only):
        self.df = df
        self.transforms = transforms
        self.unique_label = unique_label
        self.unique_only = unique_only
        self.centercrop = centercrop
        if self.centercrop:
            self.images = laod_pkl('../input/bengaliai-cv19/train_images.pkl')
        else:
            self.images = prepare_image(data_type='train')
        self.n_grapheme = 168
        self.n_vowel = 11
        self.n_consonant = 7
        self.n_unique = 1292
        if self.unique_label:
            self.n_total = self.n_grapheme + self.n_vowel + self.n_consonant + self.n_unique
        elif self.unique_only:
            self.n_total = self.n_unique
        else:
            self.n_total = self.n_grapheme + self.n_vowel + self.n_consonant

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_idx = row.original_index
        label_grapheme = row.grapheme_root
        label_vawel = row.vowel_diacritic
        label_consonant = row.consonant_diacritic
        label_unique = row.unique_label
        label = np.zeros(self.n_total).astype('f')
        if self.unique_only:
            label[label_unique] = 1
        else:
            label[label_grapheme] = 1
            label[self.n_grapheme + label_vawel] = 1
            label[self.n_grapheme + self.n_vowel + label_consonant] = 1
            if self.unique_label:
                label[self.n_grapheme + self.n_vowel + self.n_consonant + label_unique] = 1

        img = self.images[image_idx,:,:]
        if self.centercrop:
            img = img.astype(np.float32) / 255. # already inversed
        else:
            img = (255 - img).astype(np.float32) / 255.
        img = np.tile(img[:,:,np.newaxis],(1,1,3))
        augmented = self.transforms(image=img)
        img = augmented['image']
        return img, label

    def __len__(self):
        return len(self.df)

class TestDataset(Dataset):
    def __init__(self, config, transforms, test_img_index):
        self.transforms = transforms
        self.centercrop = config.data.centercrop
        if self.centercrop:
            self.images = prepare_pkl_image(indices=[test_img_index])
        else:
            self.images = prepare_image(data_type='test', submission=True, indices=[test_img_index])

    def __getitem__(self, idx):
        img = self.images[idx,:,:]
        if self.centercrop:
            img = img.astype(np.float32) / 255. # already inversed
        else:
            img = (255 - img).astype(np.float32) / 255.
        img = np.tile(img[:,:,np.newaxis],(1,1,3))
        augmented = self.transforms(image=img)
        img = augmented['image']
        return img, idx

    def __len__(self):
        return len(self.images)

def get_loader(phase, config, idx_fold=-1, test_img_index=0):
    if phase == 'test':
        transforms = get_transforms(config.transforms.test)
        image_dataset = TestDataset(config, transforms, test_img_index)
        batch_size = config.test.batch_size
        is_shuffle = False
        drop_last = False

    else:  # train or valid
        if os.path.exists('data/folds.csv'):
            df = pd.read_csv('data/folds.csv')
        else:
            raise Exception('You need to run split_folds.py beforehand.')

        if phase == "train":
            df = df[df['folds'] != idx_fold]
            transforms=get_transforms(config.transforms.train)
            batch_size = config.train.batch_size
            is_shuffle = True
            drop_last = True
        else:
            df = df[df['folds'] == idx_fold]
            transforms = get_transforms(config.transforms.test)
            batch_size = config.test.batch_size
            is_shuffle = False
            drop_last = False

        image_dataset = TrainDataset(df, config, transforms, unique_label=config.data.unique_label, unique_only=config.data.unique_only)

    return DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=is_shuffle,
        drop_last=drop_last,
    )
