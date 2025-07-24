import torch
import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))
import numpy as np
import torchvision.transforms as transforms
import glob
import cv2
from PIL import Image
import pandas as pd
import scprep as scp
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import torchvision.transforms.functional as TF
import random
import scanpy as sc
import warnings

warnings.filterwarnings("ignore")




class HERDataset(torch.utils.data.Dataset):
    def __init__(self, train=True, gene_list=None, ds=None, fold=0):
        super().__init__()
        self.cnt_dir = 'data/her2st/raw_data'
        self.img_dir = 'data/her2st/ST-imgs'
        self.pos_dir = 'data/her2st/ST-spotfiles'
        self.lbl_dir = 'data/her2st/ST-pat'
        self.r = 224 // 2
        gene_list = list(np.load('data/her2st/genes_her2st_HEG.npy', allow_pickle=True)[:, 0])[0:250]
        self.gene_list = gene_list

        patients = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        names = os.listdir(self.cnt_dir)
        names.sort()
        names = [i[:2] for i in names]  # ['A1', 'A2', 'A3',..]

        self.train = train

        te_names = [i for i in names if patients[fold] in i]
        tr_names = list(set(names) - set(te_names))
        if train:
            names = tr_names
        else:
            names = te_names
            self.meta_dict = {i: self.get_meta(i) for i in names}
            self.names = te_names
            self.label = {i: None for i in self.names}
            self.lbl2id = {
                'invasive cancer': 0, 'breast glands': 1, 'immune infiltrate': 2,
                'cancer in situ': 3, 'connective tissue': 4, 'adipose tissue': 5, 'undetermined': -1
            }
            if not train and len(list(set(self.names) & set(['B1', 'C1', 'D1', 'E1', 'F1', 'G2']))) > 0:
                intersec = list(set(self.names) & set(['B1', 'C1', 'D1', 'E1', 'F1', 'G2']))
                self.lbl_dict = {i: self.get_lbl(i) for i in intersec}
                idx = self.meta_dict[intersec[0]].index
                lbl = self.lbl_dict[intersec[0]]
                lbl = lbl.loc[idx, :]['label'].values
                self.label[intersec[0]] = lbl

            # if not train and self.names[0] in ['B1', 'C1', 'D1', 'E1', 'F1', 'G2']:
            #     self.lbl_dict = {i: self.get_lbl(i) for i in self.names}
            #     idx = self.meta_dict[self.names[0]].index
            #     lbl = self.lbl_dict[self.names[0]]
            #     lbl = lbl.loc[idx, :]['label'].values
            #     self.label[self.names[0]] = lbl

        # print("Loading imgs ...")
        self.img_dict = {i: self.get_img(i) for i in names}
        # print("Loading metadata...")
        self.meta_dict = {i: self.get_meta(i) for i in names}

        self.gene_set = list(gene_list)
        self.exp_dict = {i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i, m in
                         self.meta_dict.items()}
        # self.exp_dict = {i: m([self.gene_set].values) for i, m in self.meta_dict.items()}
        # print(self.exp_dict)
        self.center_dict = {i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int) for i, m in
                            self.meta_dict.items()}

        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}

        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(names))

        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        i = 0
        item = {}
        while index >= self.cumlen[i]:
            i += 1
        idx = index
        if i > 0:
            idx = index - self.cumlen[i - 1]
        exp = self.exp_dict[self.id2name[i]][idx]
        center = self.center_dict[self.id2name[i]][idx]
        loc = self.loc_dict[self.id2name[i]][idx]

        exp = torch.Tensor(exp)
        loc = torch.Tensor(loc)

        x, y = center
        patch = self.img_dict[self.id2name[i]].crop((x - self.r, y - self.r, x + self.r, y + self.r))
        if self.train:
            patch = self.transforms(patch)
        else:
            patch = transforms.ToTensor()(patch)
        if self.train:
            item["image"] = patch
            item["position"] = loc
            item["expression"] = exp
            return item

        else:
            item["image"] = patch
            item["position"] = loc
            item["expression"] = exp
            item["center"] = torch.Tensor(center)
            return item

    def __len__(self):
        return self.cumlen[-1]

    def get_meta(self, name, gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        # print(pos)
        meta = cnt.join((pos.set_index('id')))

        return meta

    def get_pos(self, name):
        path = self.pos_dir + '/' + name + '_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path, sep='\t')
        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id

        return df

    def get_cnt(self, name):
        path = self.cnt_dir + '/' + name + '.tsv'

        df = pd.read_csv(path, sep='\t', index_col=0)
        return df

    def get_img(self, name):
        pre = self.img_dir + '/' + name[0] + '/' + name  # data/her2st/data/ST-imgs/D/D6
        fig_name = os.listdir(pre)[0]
        path = pre + '/' + fig_name
        im = Image.open(path)
        return im

    def get_lbl(self, name):
        path = self.lbl_dir + '/' + 'lbl' + '/' + name + '_labeled_coordinates.tsv'
        df = pd.read_csv(path, sep='\t')
        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)
        df.set_index('id', inplace=True)

        return df

    def get_overlap(self, meta_dict, gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set & set(i.columns)
        return list(gene_set)

