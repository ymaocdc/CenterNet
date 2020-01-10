from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .image import transform_preds
from .ddd_utils import ddd2locrot, unproject_2d_to_3d


def get_pred_depth(depth):
  return depth

def get_alpha(rot):
  # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
  #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
  # return rot[:, 0]
  idx = rot[:, 1] > rot[:, 5]
  # alpha1 = np.arctan(rot[:, 2] / rot[:, 3])
  alpha1 = np.arctan2(rot[:, 2], rot[:, 3])
  alpha1 = np.remainder(alpha1+np.pi*2, np.pi*2)
  alpha2 = np.arctan2(rot[:, 6], rot[:, 7]) + np.pi
  # alpha2 = np.arctan(rot[:, 6] / rot[:, 7])+np.pi
  return alpha1 * idx + alpha2 * (1 - idx)

import math
def quaternion_to_euler_angle(q):
  """Convert quaternion to euler angel.
  Input:
      q: 1 * 4 vector,
  Output:
      angle: 1 x 3 vector, each row is [roll, pitch, yaw]
  """
  w, x, y, z = q
  t0 = +2.0 * (w * x + y * z)
  t1 = +1.0 - 2.0 * (x * x + y * y)
  X = math.atan2(t0, t1)

  t2 = +2.0 * (w * y - z * x)
  t2 = +1.0 if t2 > +1.0 else t2
  t2 = -1.0 if t2 < -1.0 else t2
  Y = math.asin(t2)

  t3 = +2.0 * (w * z + x * y)
  t4 = +1.0 - 2.0 * (y * y + z * z)
  Z = math.atan2(t3, t4)

  return np.array([X, Y, Z], dtype=np.float32)



def ddd_post_process_2d(dets, c, s, opt):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  ret = []
  include_wh = 1
  for i in range(dets.shape[0]):
    top_preds = {}

    xymin = dets[i, :, :2] - dets[i, :, 15:17]/2
    xymax = dets[i, :, :2] + dets[i, :, 15:17]/2

    dets[i, :, :2] = transform_preds(
          xymin, c[i], s[i], (opt.output_w, opt.output_h))
    dets[i, :, 15:17] = transform_preds(
      xymax, c[i], s[i], (opt.output_w, opt.output_h))
    dets[i, :, 18:20] = transform_preds(
      dets[i, :, 18:20], c[i], s[i], (opt.output_w, opt.output_h))

    if opt.crop_half:
      dets[i, :, 1] += opt.crop_from
      dets[i, :, 16] += opt.crop_from
      dets[i, :, 19] += opt.crop_from

    classes = dets[i, :, 17]
    for j in range(opt.num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :3].astype(np.float32),
        dets[i, inds, 11:12].astype(np.float32),
        dets[i, inds, 12:15].astype(np.float32)], axis=1)
      if include_wh:
        top_preds[j + 1] = np.concatenate([
          top_preds[j + 1],
          dets[i, inds, 15:17].astype(np.float32)], axis=1)
      if opt.reg_3d_center:
        top_preds[j + 1] = np.concatenate(
          [top_preds[j + 1], dets[i, inds, 18:20].astype(np.float32)], axis=1)
      else:
        top_preds[j + 1] = np.concatenate(
          [top_preds[j + 1], np.zeros((top_preds[j + 1].shape[0], 2), dtype=np.float32)], axis=1)

      if opt.reg_q:
        top_preds[j + 1] = np.concatenate([top_preds[j + 1], dets[i, inds, 20:24].astype(np.float32)], axis=1)
      else:
        top_preds[j + 1] = np.concatenate(
          [top_preds[j + 1], np.zeros((top_preds[j + 1].shape[0], 4), dtype=np.float32)], axis=1)
    ret.append(top_preds)
  return ret

def ddd_post_process_3d(dets, calibs, opt):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  ret = []
  for i in range(len(dets)):
    preds = {}
    for cls_ind in dets[i].keys():
      preds[cls_ind] = []
      for j in range(len(dets[i][cls_ind])):
        center = (dets[i][cls_ind][j][:2] + dets[i][cls_ind][j][7:9])/2

        score = dets[i][cls_ind][j][2]
        depth = dets[i][cls_ind][j][3]
        dimensions = dets[i][cls_ind][j][4:7]
        wh = -dets[i][cls_ind][j][:2] + dets[i][cls_ind][j][7:9]

        if opt.reg_3d_center:
          center3d = dets[i][cls_ind][j][9:11]
          locations = unproject_2d_to_3d(center3d, depth, calibs[0])
        else:
          locations = unproject_2d_to_3d(center, depth, calibs[0])

        if opt.reg_q:
          quaternions = dets[i][cls_ind][j][11:15]
          norm = np.linalg.norm(quaternions, axis=0)
          rot_pred_norm = quaternions / norm
          # normalise the unit quaternion here
          euler_angle = quaternion_to_euler_angle(rot_pred_norm)
        else:
          euler_angle = np.array([0,0,0], dtype=np.float)
        bbox = [center[0] - wh[0] / 2, center[1] - wh[1] / 2,
                center[0] + wh[0] / 2, center[1] + wh[1] / 2]

        pred = bbox + dimensions.tolist() + \
               locations.tolist() + euler_angle.tolist() + [score]

        if opt.reg_3d_center:
          pred = pred + center3d.tolist()
        else:
          pred = pred + [None, None]

        preds[cls_ind].append(pred)
      preds[cls_ind] = np.array(preds[cls_ind], dtype=np.float32)
    ret.append(preds)
  return ret

def ddd_post_process(dets, c, s, calibs, opt):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  dets = ddd_post_process_2d(dets, c, s, opt)
  dets = ddd_post_process_3d(dets, calibs, opt)
  return dets


def ctdet_post_process(dets, c, s, h, w, num_classes):
  # dets: batch x max_dets x dim
  # return 1-based class det dict
  ret = []
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (w, h))
    dets[i, :, 2:4] = transform_preds(
          dets[i, :, 2:4], c[i], s[i], (w, h))
    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :4].astype(np.float32),
        dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret


def multi_pose_post_process(dets, c, s, h, w):
  # dets: batch x max_dets x 40
  # return list of 39 in image coord
  ret = []
  for i in range(dets.shape[0]):
    bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
    pts = transform_preds(dets[i, :, 5:39].reshape(-1, 2), c[i], s[i], (w, h))
    top_preds = np.concatenate(
      [bbox.reshape(-1, 4), dets[i, :, 4:5], 
       pts.reshape(-1, 34)], axis=1).astype(np.float32).tolist()
    ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
  return ret
