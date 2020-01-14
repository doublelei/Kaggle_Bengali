import glob
import math
import random

import cv2
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

sz = 128
bs = 128
nfolds = 4 #keep the same split as the initial dataset
fold = 0
SEED = 2019
TRAIN = '../data/grapheme-imgs-128x128/'
LABELS = '../input/bengaliai-cv19/train.csv'
arch = models.densenet121

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)


# def transform_csi_base(csi, name, mean, mean_type, var):
#     """
#     The transformer for csi data
#     :param csi: tensor, float64, shape: (number of consecutive csi data * csi sub-carriers * Rx * Tx）
#     :param name: str, the scene name which this csi data belongs to, like 2019_10_10_0
#     :param mean: dict, the mean of each scene. If not reduce mean, mean=None
#     :param mean_type: str, the mean type to reduce
#     :param var: dict, the variance of each scene. If not add noise, var=None
#     :return: csi: tensor, float64, shape: ((number of consecutive csi data * csi sub-carriers) * Rx * Tx）
#     """
#     if mean:
#         if mean_type == 'mean':
#             csi -= mean[name]
#         elif mean_type == 'geo_mean':
#             csi /= (mean[name] + 1e-8)
#     if var:
#         csi += np.random.uniform(-1, 1, size=(5, 30, 3, 3)) * np.sqrt(var[name])
#     csi = torch.from_numpy(csi).type(torch.FloatTensor).view(-1, 3, 3)
#     return csi


# def transform_ann_base(ann, w, h, max_persons=5):
#     """
#     The transformer for annotations
#     :param ann: dict, the raw input including mask, boxes, (edges and joints in the future)
#                 ann['mask'], ndarray, uint8, shape=(original height, original width)
#                 ann['boxes'], ndarray, float32, shape=(number of persons, 5), second dims is in (x, y, w, h, confidence)
#     :param w: int, output width
#     :param h: int, output height
#     :param max_persons: int, the max number of person in all inputs
#     :return: ann: dict, transformed annotations
#                   ann['mask_hm']: tensor, int64, shape=(1, output height, output width)
#                   ann['bbox_ct_hm']: tensor, int64, shape=(1, output height, output width), center heatmap for bbox
#                   ann['bbox_wh']: tensor, int64, shape=(max persons, 2), bounding box width and height
#                   ann['bbox_ind']: tensor, int64, shape=(max persons), bounding box indexes
#                   ann['bbox_offset']: tensor, float32, shape=(max persons, 2), bounding box center offset
#                   ann['bbox_mask']: tensor, int64, shape=(max persons), bounding box mask
#                   ann['boxes']: ndarray, float32, shape=(max persons, 5)
#     """

#     # resize mask
#     mask = ann['mask']
#     o_h, o_w = mask.shape
#     mask = cv2.resize(mask, (w, h))
#     mask = torch.from_numpy(mask).type(torch.LongTensor).view(1, h, w)

#     # resize bounding box
#     boxes = ann['boxes']
#     boxes[:, [0, 2]] /= (o_w / w)
#     boxes[:, [1, 3]] /= (o_h / h)

#     # transform bbox
#     bbox_ct_hm = np.zeros((1, h, w), dtype=np.float32)
#     bbox_wh = np.zeros((max_persons, 2), dtype=np.float32)
#     bbox_ind = np.zeros(max_persons, dtype=np.int64)
#     bbox_offset = np.zeros((max_persons, 2), dtype=np.float32)
#     bbox_mask = np.zeros(max_persons, dtype=np.uint8)
#     if len(boxes > max_persons):
#         boxes = boxes[:max_persons]
#     for k, bbox in enumerate(boxes):
#         bb_w, bb_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
#         radius = max(0, int(gaussian_radius((math.ceil(bb_h), math.ceil(bb_w)))))
#         center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
#         center_int = center.astype(np.int32)
#         bbox_wh[k] = 1. * bb_w, 1. * bb_h
#         bbox_ind[k] = center_int[1] * w + center_int[0]
#         bbox_offset[k] = center - center_int
#         bbox_mask[k] = 1
#         draw_umich_gaussian(bbox_ct_hm[0], center_int, radius)
#     boxes = np.pad(boxes, ((0, max_persons - len(boxes)), (0, 0)), 'constant', constant_values=(0, 0))

#     return {'mask_hm': mask, 'bbox_ct_hm': torch.from_numpy(bbox_ct_hm), 'bbox_wh': torch.from_numpy(bbox_wh),
#             'bbox_ind': torch.from_numpy(bbox_ind), 'bbox_offset': torch.from_numpy(bbox_offset),
#             'bbox_mask': torch.from_numpy(bbox_mask),
#             'boxes': boxes.astype(np.float32)}


class BengaliDataset(Dataset):
    """Bengali dataset"""

    def __init__(self, data_files, mean, var, opt, transform_csi=transform_csi_base, transform_ann=transform_ann_base):
        """
        :param data_files: list, the paths of all input data files
        :param mean: mean: dict, the mean of each scene. If not reduce mean, mean=None
        :param var: var: dict, the variance of each scene. If not add noise, var=None
        :param opt: opt, configurations for the model
        :param transform_csi: function, the transformer for csi
        :param transform_ann: function, the transformer for annotations
        """
        self.data_files = data_files
        self.transform_csi = transform_csi
        self.transform_ann = transform_ann
        self.opt = opt
        self.mean = mean
        self.var = var

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        name = self.data_files[idx].split('/')[-2]
        data = loadmat(self.data_files[idx])

        csi = self.transform_csi(data['csi_serial'], name, self.mean, self.opt.mean_type, self.var)

        if len(data) != 1:
            inp = self.transform_ann(data, self.opt.output_w, self.opt.output_h, self.opt.max_persons)
        else:
            # return zeros-like tensor like it's a background file without persons
            inp = {'mask_hm': torch.zeros((1, self.opt.output_h, self.opt.output_w), dtype=torch.long),
                   'bbox_ct_hm': torch.zeros((1, self.opt.output_h, self.opt.output_w), dtype=torch.float32),
                   'bbox_wh': torch.zeros((self.opt.max_persons, 2), dtype=torch.float32),
                   'bbox_ind': torch.zeros(self.opt.max_persons, dtype=torch.int64),
                   'bbox_offset': torch.zeros((self.opt.max_persons, 2), dtype=torch.float32),
                   'bbox_mask': torch.zeros(self.opt.max_persons, dtype=torch.uint8),
                   'boxes': np.zeros((self.opt.max_persons, 5), dtype=np.float32)}

        if self.opt.with_bg:
            if self.opt.bg_type == 'mean' or self.opt.bg_type == 'geo_mean':
                inp['bg'] = torch.from_numpy(self.bg[name]).type(torch.FloatTensor).view(-1, 3, 3)
            elif self.opt.bg_type == 'random':
                inp['bg'] = torch.from_numpy(np.abs(loadmat(random.choice(self.bg[name], 1))['csi_serial'])) \
                    .type(torch.FloatTensor).view(-1, 3, 3)

        inp['csi'] = csi
        return inp


def generate_dataset(data_paths, opt):
    """
    generate dataset according to opt and data paths
    :param data_paths: the paths to store data for all scenes
    :param opt: opt, configurations for the model
    :return:
    """
    data_mats, mean, var, bg = [], {}, {}, {}
    random.seed(0)

    for path in data_paths:
        name = path.split('/')[-1]
        data_mats.extend(glob.glob('{}/*_*.mat'.format(path)))
        if opt.reduce_mean:
            mean_path = path.replace('csi_data', 'background/{}'.format(opt.mean_type)) + '.npy'
            mean[name] = np.tile(np.load(mean_path), (5, 1, 1, 1))
        if opt.add_noise:
            var_path = path.replace('csi_data', 'background/vars') + '.npy'
            var[name] = np.load(var_path)
        if opt.with_bg:
            if opt.bg_type == 'random':
                bg[name] = glob.glob('{}/*_0.mat'.format(path))
            else:
                bg_path = path.replace('csi_data', 'background/{}'.format(opt.bg_type)) + '.npy'
                bg[name] = np.tile(np.load(bg_path), (5, 1, 1, 1))

    if opt.leave_one_scene_out:
        data_loader = DataLoader(WiFiSensingDataset(data_mats, mean, var, opt), batch_size=opt.batch_szie,
                                 shuffle=True, pin_memory=True, drop_last=True, num_workers=opt.num_workers)
        return data_loader
    else:
        random.shuffle(data_mats)
        train_mats = data_mats[:int(len(data_mats) * (1 - opt.testing_rate))]
        test_mats = data_mats[int(len(data_mats) * (1 - opt.testing_rate)):]

        assert len(set(train_mats) - set(test_mats)) == len(train_mats)
        train_data_loader = DataLoader(WiFiSensingDataset(train_mats, mean, var, opt), batch_size=opt.batch_size,
                                       shuffle=True, pin_memory=True, drop_last=True, num_workers=opt.num_workers)
        test_data_loader = DataLoader(WiFiSensingDataset(test_mats, mean, var, opt), batch_size=opt.batch_size,
                                      shuffle=True, pin_memory=True, drop_last=True, num_workers=opt.num_workers)
        return train_data_loader, test_data_loader