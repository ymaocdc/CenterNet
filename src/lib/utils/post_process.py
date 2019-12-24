from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .image import transform_preds
from .ddd_utils import ddd2locrot


def get_pred_depth(depth):
  return depth

def get_alpha(rot):
  # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
  #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
  # return rot[:, 0]
  idx = rot[:, 1] > rot[:, 5]
  alpha1 = np.arctan(rot[:, 2] / rot[:, 3])
  alpha2 = np.arctan(rot[:, 6] / rot[:, 7])+np.pi
  return alpha1 * idx + alpha2 * (1 - idx)
  

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
    dets[i, :, 19:21] = transform_preds(
      dets[i, :, 19:21], c[i], s[i], (opt.output_w, opt.output_h))

    lbpe = transform_preds(
        np.array([dets[i, :, 21], xymax[:, 1]], dtype=np.float).transpose(), c[i], s[i], (opt.output_w, opt.output_h))
    rbpe = transform_preds(
        np.array([dets[i, :, 22], xymax[:, 1]], dtype=np.float).transpose(), c[i], s[i], (opt.output_w, opt.output_h))
    dets[i, :, 21] = lbpe[:, 0]
    dets[i, :, 22] = rbpe[:, 0]

    lfpe = transform_preds(
      np.array([dets[i, :, 23], xymax[:, 1]], dtype=np.float).transpose(), c[i], s[i], (opt.output_w, opt.output_h))
    rfpe = transform_preds(
      np.array([dets[i, :, 24], xymax[:, 1]], dtype=np.float).transpose(), c[i], s[i], (opt.output_w, opt.output_h))
    dets[i, :, 23] = lfpe[:, 0]
    dets[i, :, 24] = rfpe[:, 0]


    classes = dets[i, :, 17]
    for j in range(opt.num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :3].astype(np.float32),
        get_alpha(dets[i, inds, 3:11])[:, np.newaxis].astype(np.float32),
        get_pred_depth(dets[i, inds, 11:12]).astype(np.float32),
        dets[i, inds, 12:15].astype(np.float32)], axis=1)
      if include_wh:
        top_preds[j + 1] = np.concatenate([
          top_preds[j + 1],
          dets[i, inds, 15:17].astype(np.float32)], axis=1)
      if opt.reg_pitch:
        top_preds[j + 1] = np.concatenate([top_preds[j + 1], dets[i, inds, 18][:, np.newaxis].astype(np.float32)], axis=1)
      else:
        top_preds[j + 1] = np.concatenate(
          [top_preds[j + 1], np.zeros((top_preds[j + 1].shape[0], 1), dtype=np.float32)], axis=1)

      if opt.reg_3d_center:
        top_preds[j + 1] = np.concatenate(
          [top_preds[j + 1], dets[i, inds, 19:21].astype(np.float32)], axis=1)
      else:
        top_preds[j + 1] = np.concatenate(
          [top_preds[j + 1], np.zeros((top_preds[j + 1].shape[0], 2), dtype=np.float32)], axis=1)


      if opt.reg_BPE:
        top_preds[j + 1] = np.concatenate(
          [top_preds[j + 1], dets[i, inds, 21:23].astype(np.float32)], axis=1)
      else:
        top_preds[j + 1] = np.concatenate(
          [top_preds[j + 1], np.zeros((top_preds[j + 1].shape[0], 2), dtype=np.float32)], axis=1)

      if opt.reg_FPE:
        top_preds[j + 1] = np.concatenate(
          [top_preds[j + 1], dets[i, inds, 23:25].astype(np.float32)], axis=1)
      else:
        top_preds[j + 1] = np.concatenate(
          [top_preds[j + 1], np.zeros((top_preds[j + 1].shape[0], 2), dtype=np.float32)], axis=1)

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
        center = (dets[i][cls_ind][j][:2] + dets[i][cls_ind][j][8:10])/2

        score = dets[i][cls_ind][j][2]
        alpha = dets[i][cls_ind][j][3]
        depth = dets[i][cls_ind][j][4]
        dimensions = dets[i][cls_ind][j][5:8]
        wh = -dets[i][cls_ind][j][:2] + dets[i][cls_ind][j][8:10]
        if opt.reg_pitch:
          pitch = dets[i][cls_ind][j][10]
        if opt.reg_3d_center:
          center3d = dets[i][cls_ind][j][11:13]
          locations, rotation_y = ddd2locrot(
            center3d, alpha, dimensions, depth, calibs[0])
        else:
          locations, rotation_y = ddd2locrot(
            center, alpha, dimensions, depth, calibs[0])
        bbox = [center[0] - wh[0] / 2, center[1] - wh[1] / 2,
                center[0] + wh[0] / 2, center[1] + wh[1] / 2]
        pred = [alpha] + bbox + dimensions.tolist() + \
               locations.tolist() + [rotation_y, score]
        if opt.reg_pitch:
          pred = pred + [pitch]
        else:
          pred = pred + [None]

        if opt.reg_BPE:
          BPE = dets[i][cls_ind][j][13:15]
          pred = pred + BPE.tolist()
        else:
          pred = pred + [None]
        if opt.reg_FPE:
          FPE = dets[i][cls_ind][j][15:17]
          pred = pred + FPE.tolist()
        else:
          pred = pred + [None]

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
