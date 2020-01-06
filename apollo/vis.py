import os, glob, cv2
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import os.path as osp


if __name__ == '__main__':
    root_dir = '/Users/yunxiangmao/work/apollo/data/train'
    image_dir = osp.join(root_dir, 'images')
    keypoint_dir = osp.join(root_dir, 'keypoints')
    pose_dir = osp.join(root_dir, 'car_poses')
