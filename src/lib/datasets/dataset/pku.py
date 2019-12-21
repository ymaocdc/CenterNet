from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import pycocotools.coco as coco
import numpy as np
import torch
import json
import cv2
import os
import math
import pandas as pd
import torch.utils.data as data


class PKU(data.Dataset):
    num_classes = 3
    default_resolution = [512, 512]
    mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(PKU, self).__init__()
        self.data_dir = os.path.join(opt.data_dir)
        self.root_dir = opt.root_dir
        # self.img_dir = os.path.join(self.data_dir, 'images')
        # if opt.trainval:
        #     split = 'trainval' if split == 'train' else 'test'
        #     self.img_dir = os.path.join(self.data_dir, 'images', split)
        #     self.annot_path = os.path.join(
        #         self.data_dir, 'annotations', 'kitti_{}.json').format(split)
        # else:
        #     self.annot_path = os.path.join(self.data_dir,
        #                                    'annotations', 'kitti_{}_{}.json').format(opt.kitti_split, split)

        # split =
        self.img_dir = os.path.join(self.data_dir, 'images', '{}_images'.format(split))
        self.annot_path = os.path.join(self.data_dir, 'annotations', '{}_coco_format.json').format(split)

        # self.annot_path = os.path.join(self.root_dir, 'val_coco_format_overfit.json')
        self.max_objs = 50
        self.class_name = [
            '__background__', '2x', '3x', 'SUV']

        self._valid_ids = np.arange(1, len(self.class_name), dtype=np.int32)
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}

        # self.cat_ids = {1: 0, 2: 1, 3: 2, 4: -3, 5: -3, 6: -2, 7: -99, 8: -99, 9: -1}

        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        self.split = split
        self.opt = opt
        self.alpha_in_degree = False

        print('==> initializing pku {} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)
        self.calib = np.array([[2304.5479, 0,  1686.2379],
           [0, 2305.8757, 1354.9849],
           [0, 0, 1]], dtype=np.float32)

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def __len__(self):
        return self.num_samples

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        pass

    def coords2str(self, coords, names=['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence']):
        s = []
        for c in coords:
            s.append(str(c))
        return ' '.join(s)

    def save_results(self, results, save_dir):



        predictions = []

        for img_id in results.keys():
            for cls_ind in results[img_id]:
                for j in range(len(results[img_id][cls_ind])):
                    class_name = self.class_name[cls_ind]
                    s = [results[img_id][cls_ind][j][11], -0.1, 0, results[img_id][cls_ind][j][8:11], results[img_id][cls_ind][j][12]]
                    predictions.append(self.coords2str(s))

        test = pd.read_csv( os.path.join(self.root_dir, 'sample_submission.csv'))
        test['PredictionString'] = predictions
        test.to_csv(os.path.join(self.root_dir, 'submission_trial.csv'), index=False)
        test.head()




    def run_eval(self, results, save_dir):
        self.save_results(results, save_dir)


