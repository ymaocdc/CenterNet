from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os, glob
import cv2
import pandas as pd
from opts import opts
import numpy as np
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


def coords2str(coords, names=['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence']):
    s = []
    for c in coords:
        s.append(str(c))
    return ' '.join(s)

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  # opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)

  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    detector.pause = False
    while True:
        _, img = cam.read()
        cv2.imshow('input', img)
        ret = detector.run(img)
        time_str = ''
        for stat in time_stats:
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)
        if cv2.waitKey(1) == 27:
            return  # esc to quit
  else:
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]
    test_masks_dir = '/xmotors_ai_shared/datasets/incubator/user/yus/dataset/pku/test_masks'
    test_masks_files = glob.glob(os.path.join(test_masks_dir, '*jpg'))
    predictions = {}
    for (image_name) in image_names:
        ret = detector.run(image_name)
        time_str = ''
        for stat in time_stats:
            time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)
        pid = image_name.split('/')[-1].split('.j')[0]
        if os.path.exists(os.path.join(test_masks_dir, pid+'.jpg')):
            test_mask = cv2.imread(os.path.join(test_masks_dir, pid+'.jpg'), 0)


        ret = ret['results']
        predictions[image_name.split('/')[-1].split('.j')[0]] = ''
        for cls_ind in ret:
            for j in range(len(ret[cls_ind])):
                bbox = ret[cls_ind][j][1:5]
                xc = int((bbox[0] +bbox[2])//2)
                yc = int((bbox[1] + bbox[3])//2)
                if test_mask[yc, xc] > 125:
                    continue
                yaw = ret[cls_ind][j][11]
                pitch = ret[cls_ind][j][13]



                s = [pitch, -yaw, -np.pi, ret[cls_ind][j][8], ret[cls_ind][j][9], ret[cls_ind][j][10],
                     ret[cls_ind][j][12]]
                predictions[image_name.split('/')[-1].split('.j')[0]] += ' '
                predictions[image_name.split('/')[-1].split('.j')[0]] += coords2str(s)

    test = pd.read_csv(os.path.join(opt.root_dir, 'sample_submission.csv'))
    for idx, image_id in enumerate(test['ImageId']):
        test['PredictionString'][idx] = predictions[image_id]

    test.to_csv(os.path.join(opt.root_dir, 'submission_regALL.csv'), index=False)
    test.head()

    # for (image_name) in image_names:
    #   ret = detector.run(image_name)
    #   time_str = ''
    #   for stat in time_stats:
    #     time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
    #   print(time_str)
if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
