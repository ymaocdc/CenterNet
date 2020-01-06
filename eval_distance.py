import os, json, glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def bbox_overlaps(box1, box2):
  area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
  area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
  inter = max(min(box1[2], box2[2]) - max(box1[0], box2[0]) + 1, 0) * \
          max(min(box1[3], box2[3]) - max(box1[1], box2[1]) + 1, 0)
  iou = 1.0 * inter / (area1 + area2 - inter)
  return iou
def coords2str(coords, names=['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence']):
    s = []
    for c in coords:
        s.append(str(c))
    return ' '.join(s)



mode = 'train'
root_folder = '/xmotors_ai_shared/datasets/incubator/user/yus/dataset/pku'
image_folder = os.path.join(root_folder, 'data/images', '{}_images'.format(mode))
label_folder = os.path.join(root_folder, 'with_mask_labels')

pred_folder = '/xmotors_ai_shared/datasets/incubator/user/yus/dataset/pku/model_prediction_resutls/pku_everything_equalweight_5121024_flip_training'

__CLASS__ = ['__background__', '2x', '3x', 'SUV']
_class_ = np.array(__CLASS__)

label_files = glob.glob(os.path.join(label_folder, '*.json'))
label_files.sort()
pred_files = glob.glob(os.path.join(pred_folder, '*.json'))
pred_files.sort()


gt_dis = []
pred_dis = []
image_ids = []
val_pred = []
ratios = []
for image_id, label_file in enumerate(tqdm(label_files[:4000])):
    pred_file = os.path.join(pred_folder, label_file.split('/')[-1])
    if not os.path.exists(pred_file):
        continue

    with open(pred_file, 'r') as fin:
        pred = json.load(fin)

    with open(label_file, 'r') as fin:
        gt = json.load(fin)

    # for obj_idx, gt_obj in enumerate(gt['objects']):
        # box1 = gt_obj['2D_bbox_xyxy']
        # max_iou = 0
        # idx = -1
        # for pred_idx, pred_obj in enumerate(pred['objects']):
        #     box2 = pred_obj['2D_bbox_xyxy']
        #     iou = bbox_overlaps(box1, box2)
        #     if iou>max_iou:
        #         max_iou = iou
        #         idx = pred_idx
        # if max_iou > 0.8:
        #     pred_obj = pred['objects'][idx]
        #     gt_dis.append(gt_obj['3D_location'][:3])
        #     pred_dis.append(pred_obj['3D_location'])

    tmp_pred = []
    for pred_idx, pred_obj in enumerate(pred['objects']):
        box1 = pred_obj['2D_bbox_xyxy']
        gt_loc = None
        max_iou = 0
        for obj_idx, gt_obj in enumerate(gt['objects']):
            box2 = gt_obj['2D_bbox_xyxy']
            iou = bbox_overlaps(box1, box2)
            if iou > max_iou:
                max_iou = iou
                idx = obj_idx
        if max_iou > 0.8:
            gt_obj = gt['objects'][idx]
            gt_loc = gt_obj['3D_location'][:3]
        if gt_loc is None:
            x,y,z = pred_obj['3D_location']
        else:
            # ratio = pred_obj['3D_location'][2]/gt_loc[2]
            # x, y, z = gt_loc[0]*ratio, gt_loc[1]*ratio, gt_loc[2]*ratio
            ratio = gt_loc[2]/pred_obj['3D_location'][2]
            x, y, z = pred_obj['3D_location'][0] * ratio, pred_obj['3D_location'][1] * ratio, pred_obj['3D_location'][2] * ratio
            ratios.append(ratio)

        box2 = pred_obj['2D_bbox_xyxy']
        pitch = pred_obj['pitch']
        # roll= pred_obj['roll']
        yaw = pred_obj['global_yaw']
        score = pred_obj['conf_score']
        s = [pitch, -yaw, -np.pi, x,y,z, score]
        tmp_pred.append(coords2str(s))
    tmp_pred = ' '.join(tmp_pred)

    val_pred.append(tmp_pred)
    image_ids.append(label_file.split('/')[-1].split('.js')[0])

data = {
        'ImageId': image_ids,
        'PredictionString': val_pred,
    }
val_df = pd.DataFrame(data=data)
root_dir = '/xmotors_ai_shared/datasets/incubator/user/yus/dataset/pku'
save_to = os.path.join(root_dir, 'submission/same_scale_pku_everything_equalweight_5121024_flip_training.csv')
val_df.to_csv(save_to, index=False)

print(save_to)

ratios = np.array(ratios)
plt.hist(ratios, normed=True, bins=200)
plt.title('ratios')
plt.show()


gt_dis = np.array(gt_dis)
pred_dis = np.array(pred_dis)

np.save('/xmotors_ai_shared/datasets/incubator/user/yus/dataset/pku/model_prediction_resutls/pku_everything_equalweight_5121024_flip_training/pred.npy', pred_dis)
np.save('/xmotors_ai_shared/datasets/incubator/user/yus/dataset/pku/model_prediction_resutls/pku_everything_equalweight_5121024_flip_training/gt.npy', gt_dis)
diff = pred_dis - gt_dis
plt.hist(diff[:,0], normed=True, bins=200)
plt.title('x diff')
plt.show()
plt.hist(diff[:,1], normed=True, bins=200)
plt.title('y diff')
plt.show()
plt.hist(diff[:,2], normed=True, bins=200)
plt.title('z diff')
plt.show()

dis_diff = np.sqrt((diff**2).sum(axis=1))
plt.hist(dis_diff, normed=True, bins=200)
plt.title('dis diff')
plt.show()

import plotly.express as px
import pandas as pd
x = gt_dis[:,0]
y = gt_dis[:,1]
z = gt_dis[:,2]
points_df = pd.DataFrame(data={'x':x, 'y':y, 'z': z})
fig = px.scatter_3d(points_df, x='x', y='y', z='z', range_x=(-50,50), range_y=(0,50), range_z=(0,250), opacity=0.1)
fig.show()

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plt.scatter(x, y, z)
plt.show()