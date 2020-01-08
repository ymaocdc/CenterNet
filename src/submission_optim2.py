from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os, glob, json
import cv2
import pandas as pd
from opts import opts
import numpy as np
from Box_roll import Box
from detectors.detector_factory import detector_factory

import matplotlib.pyplot as plt
image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def draw_lines(img, points, linewidth=3, style='allw'):

    if style == 'fb':#front back
        color = (0, 0, 255)
        cv2.line(img, tuple(points[2][:2]), tuple(points[3][:2]), color, linewidth)
        cv2.line(img, tuple(points[3][:2]), tuple(points[7][:2]), color, linewidth)
        cv2.line(img, tuple(points[6][:2]), tuple(points[7][:2]), color, linewidth)
        cv2.line(img, tuple(points[2][:2]), tuple(points[6][:2]), color, linewidth)
        color = (255, 0, 0)
        cv2.line(img, tuple(points[1][:2]), tuple(points[2][:2]), color, linewidth)
        cv2.line(img, tuple(points[3][:2]), tuple(points[4][:2]), color, linewidth)
        cv2.line(img, tuple(points[5][:2]), tuple(points[6][:2]), color, linewidth)
        cv2.line(img, tuple(points[7][:2]), tuple(points[8][:2]), color, linewidth)
        color = (0, 255, 0)
        cv2.line(img, tuple(points[1][:2]), tuple(points[5][:2]), color, linewidth)
        cv2.line(img, tuple(points[5][:2]), tuple(points[8][:2]), color, linewidth)
        cv2.line(img, tuple(points[4][:2]), tuple(points[8][:2]), color, linewidth)
        cv2.line(img, tuple(points[1][:2]), tuple(points[4][:2]), color, linewidth)
    elif style == 'tb':# top bottom
        color = (0, 0, 255)
        cv2.line(img, tuple(points[5][:2]), tuple(points[8][:2]), color, linewidth)
        cv2.line(img, tuple(points[5][:2]), tuple(points[6][:2]), color, linewidth)
        cv2.line(img, tuple(points[6][:2]), tuple(points[7][:2]), color, linewidth)
        cv2.line(img, tuple(points[7][:2]), tuple(points[8][:2]), color, linewidth)

        color = (255, 0, 0)
        cv2.line(img, tuple(points[1][:2]), tuple(points[2][:2]), color, linewidth)
        cv2.line(img, tuple(points[3][:2]), tuple(points[4][:2]), color, linewidth)
        cv2.line(img, tuple(points[2][:2]), tuple(points[3][:2]), color, linewidth)
        cv2.line(img, tuple(points[1][:2]), tuple(points[4][:2]), color, linewidth)

        color = (0, 255, 0)
        cv2.line(img, tuple(points[3][:2]), tuple(points[7][:2]), color, linewidth)
        cv2.line(img, tuple(points[2][:2]), tuple(points[6][:2]), color, linewidth)
        cv2.line(img, tuple(points[1][:2]), tuple(points[5][:2]), color, linewidth)
        cv2.line(img, tuple(points[4][:2]), tuple(points[8][:2]), color, linewidth)
    elif style == 'allw':#front back
        color = (255, 255, 255)
        cv2.line(img, tuple(points[2][:2]), tuple(points[3][:2]), color, linewidth)
        cv2.line(img, tuple(points[3][:2]), tuple(points[7][:2]), color, linewidth)
        cv2.line(img, tuple(points[6][:2]), tuple(points[7][:2]), color, linewidth)
        cv2.line(img, tuple(points[2][:2]), tuple(points[6][:2]), color, linewidth)
        # color = (255, 0, 0)
        cv2.line(img, tuple(points[1][:2]), tuple(points[2][:2]), color, linewidth)
        cv2.line(img, tuple(points[3][:2]), tuple(points[4][:2]), color, linewidth)
        cv2.line(img, tuple(points[5][:2]), tuple(points[6][:2]), color, linewidth)
        cv2.line(img, tuple(points[7][:2]), tuple(points[8][:2]), color, linewidth)
        # color = (0, 255, 0)
        cv2.line(img, tuple(points[1][:2]), tuple(points[5][:2]), color, linewidth)
        cv2.line(img, tuple(points[5][:2]), tuple(points[8][:2]), color, linewidth)
        cv2.line(img, tuple(points[4][:2]), tuple(points[8][:2]), color, linewidth)
        cv2.line(img, tuple(points[1][:2]), tuple(points[4][:2]), color, linewidth)
    else:
        assert print('wrong input style')
    return img

def draw_corners(img, points):
    "Draw 8 corners and centroid of 3d bbox on image"
    for idx, (p_x, p_y, p_z) in enumerate(points):
        if idx == 0:
            color = (0, 128, 255)
        elif idx == 1:
            color = (0, 255, 255)
        elif idx == 7:
            color = (255, 0, 0)
        else:
            color = (255, 255, 0)
        cv2.circle(img, (p_x, p_y), 7, color, -1)
    return img

class JsonSerilizable(json.JSONEncoder):
    """Helper class to help serialize dictionary"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            obj = obj.squeeze()
            _shape = obj.shape
            obj = np.array([float(_obj) for _obj in obj.flatten()]).reshape(_shape)
            return obj.tolist()
        elif isinstance(obj, list):
            return [float(_obj) for _obj in obj]
        elif isinstance(obj, str):
            pass
        elif isinstance(obj, bool):
            pass
        else:
            return float(obj)
        return json.JSONEncoder.default(self, obj)

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return super(NumpyEncoder, self).encode(bool(obj))
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def res2dict(dets, image_name, opt, center_thresh=0.2):
    typedict = {1:'2x', 2:'3x', 3:'suv'}
    result = {}
    result['file'] = image_name
    result['objects'] = []
    for cat in dets:
        for i in range(len(dets[cat])):
            if dets[cat][i, 12] > center_thresh:
                obj = {}
                dim = dets[cat][i, 5:8]
                loc  = dets[cat][i, 8:11]
                rot_y = dets[cat][i, 11]
                bbox = dets[cat][i, 1:5]
                obj['3D_dimension'] = dim
                obj['3D_location'] = loc
                obj['2D_bbox_xyxy'] = bbox
                obj['global_yaw'] = rot_y
                obj['conf_score'] = dets[cat][i, 12]
                if opt.reg_pitch:
                    obj['pitch'] = dets[cat][i, 13]
                else:
                    obj['pitch'] = None
                if opt.reg_BPE:
                    obj['BPE'] = dets[cat][i, 14:16]
                else:
                    obj['BPE'] = None
                if opt.reg_FPE:
                    obj['FPE'] = dets[cat][i, 16:18]
                else:
                    obj['FPE'] = None
                obj['type'] = typedict[cat]


                result['objects'].append(obj)
    return result

def coords2str(coords, names=['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence']):
    s = []
    for c in coords:
        s.append(str(c))
    return ' '.join(s)

def toint(x):
    return [int(i) for i in x]

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  # opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)

  suffix = 'test_results'
  model_outputfolder = os.path.join(opt.root_dir, 'model_prediction_resutls', opt.load_model.split('/')[-2]+suffix)
  if not os.path.exists(model_outputfolder):
      os.mkdir(model_outputfolder)
  if not os.path.exists(os.path.join(opt.root_dir, 'optim_resutls2')):
      os.mkdir(os.path.join(opt.root_dir, 'optim_resutls2'))


  # bbox2d_folder
  output_dir = '/xmotors_ai_shared/datasets/incubator/user/yus/dataset/pku/bbox_results/{}'.format(
      opt.load_model.split('/')[-2])
  if not os.path.exists(output_dir):
      os.mkdir(output_dir)
  opt.output_dir = output_dir

  optim_output_folder = os.path.join(opt.root_dir, 'optim_resutls2', opt.load_model.split('/')[-2]+suffix)
  if not os.path.exists(optim_output_folder):
      os.mkdir(optim_output_folder)

  test_masks_dir = '/xmotors_ai_shared/datasets/incubator/user/yus/dataset/pku/test_masks'

  use_optimization = True
  debug = True
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


    calib = np.array([[2304.5479, 0, 1686.2379],
                      [0, 2305.8757, 1354.9849],
                      [0, 0, 1]], dtype=np.float32)

    predictions = {}
    for (image_name) in image_names:
        world_im = np.zeros((2000, 600), dtype=np.uint8)
        bottom_im = np.zeros((2000, 600), dtype=np.float)
        resolution = 0.1  # m/pixel
        # if not 'ID_000372ced' in image_name:
        #     continue
        if debug:
            org_im = cv2.imread(image_name)
            img = org_im.copy()
        ret = detector.run(image_name)
        time_str = image_name
        for stat in time_stats:
            time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)
        pid = image_name.split('/')[-1].split('.j')[0]
        if os.path.exists(os.path.join(test_masks_dir, pid+'.jpg')):
            test_mask = cv2.imread(os.path.join(test_masks_dir, pid+'.jpg'), 0)
        else:
            test_mask = np.ones((2710, 3384), dtype=np.uint8)*0

        ret = ret['results']

        #save network output
        results = res2dict(ret, image_name, opt, center_thresh=0)
        with open(os.path.join(model_outputfolder, image_name.split('/')[-1].split('.')[0]+'.json'), 'w') as f_out:
            json.dump(results, f_out, indent=4, sort_keys=True, cls=NumpyEncoder)

        predictions[image_name.split('/')[-1].split('.j')[0]] = []
        for cls_ind in ret:
            for j in range(len(ret[cls_ind])):
                # j=4
                bbox = ret[cls_ind][j][1:5]
                xc = int((bbox[0] +bbox[2])//2)
                yc = int((bbox[1] + bbox[3])//2)
                if test_mask[yc, xc] > 125:
                    continue
                xmin, ymin, xmax, ymax = toint(ret[cls_ind][j, 1:5])
                bpe_l, bpe_r = toint(ret[cls_ind][j, 14:16])
                fpe_l, fpe_r = toint(ret[cls_ind][j, 16:18])
                H, W, L = ret[cls_ind][j, 5:8]
                tx, ty, tz = ret[cls_ind][j, 8:11]
                score = ret[cls_ind][j, 12]
                pitch = ret[cls_ind][j, 13]
                center3d = ret[cls_ind][j, 18:20]
                roll = ret[cls_ind][j, 20]
                try:
                    box3d = Box(xmin, ymin, xmax, ymax, bpe_l, bpe_r, fpe_l, fpe_r, calib, L, W, H, tx, ty, tz, pitch, center3d=center3d, roll=roll)
                    box3d.lift_to_3d()
                    if debug:
                        img_cor_points = box3d.img_cor_points
                        if not img_cor_points is None:
                            img_cor_points = img_cor_points
                            img = draw_corners(img, img_cor_points)
                            img = draw_lines(img, img_cor_points, style='fb')

                            text = '{0:.2f}'.format(score)
                            cv2.putText(img, text, tuple((img_cor_points[1][0] - 5, img_cor_points[1][1] - 20)), 7, 1,
                                        (0, 50, 255), 2, cv2.LINE_AA)

                            world_cor_points = box3d.world_cor_points
                            if not world_cor_points is None:
                                bottom = world_cor_points[[0, 2], :][:, [5, 6, 7, 8]]
                                bottom = bottom.transpose()
                                bottom = (bottom / resolution).astype(np.int32)
                                bottom[:, 0] = bottom[:, 0] + world_im.shape[1] // 2
                                cv2.fillConvexPoly(world_im, bottom, 255)

                    yaw = box3d.global_yaw

                except:
                    yaw = ret[cls_ind][j][11]

                if yaw is None:
                    yaw = ret[cls_ind][j][11]

                # replace the yaw and save the optimization result to json for mlp later
                ret[cls_ind][j][11] = yaw

                if yaw > np.pi:
                    yaw = yaw-np.pi*2
                if np.isnan(roll):
                    roll = -3.1
                s = [pitch, -yaw, roll, ret[cls_ind][j][8], ret[cls_ind][j][9], ret[cls_ind][j][10],
                     ret[cls_ind][j][12]]
                predictions[image_name.split('/')[-1].split('.j')[0]].append(coords2str(s))
        predictions[image_name.split('/')[-1].split('.j')[0]] = ' '.join(predictions[image_name.split('/')[-1].split('.j')[0]])

        if debug:
            # fig = plt.figure(figsize=(20, 20))
            # plt.imshow(img[:, :, ::-1])
            # plt.savefig(os.path.join(optim_output_folder, image_name.split('/')[-1].split('.')[0] + '.jpg'),  bbox_inches='tight', pad_inches=0)
            # # plt.show()
            # plt.close(fig)

            target_w = world_im.shape[1] * img.shape[0] / world_im.shape[0]
            world_im = cv2.resize(world_im, (int(target_w), img.shape[0]))
            world_im[world_im > 0] = 255
            world_im = cv2.applyColorMap(world_im, cv2.COLORMAP_JET)
            world_im = cv2.flip(world_im, 0)

            h, w = world_im.shape[:2]
            lw = 3
            for i in range(1, 20):
                row = int(h / 20 * i)
                cv2.line(world_im, (0, row), (w, row), (128, 128, 128), lw)
            for i in range(1, 6):
                col = int(w / 6 * i)
                cv2.line(world_im, (col, 0), (col, h), (128, 128, 128), lw)

            comb = np.concatenate((img[:, :, ::-1], world_im), axis=1)

            fig = plt.figure(figsize=(20, 20))
            plt.imshow(comb)
            plt.savefig(os.path.join(optim_output_folder, image_name.split('/')[-1].split('.')[0] + '.jpg'),
                        bbox_inches='tight', pad_inches=0)
            # plt.show()
            plt.close(fig)

        optm_results = res2dict(ret, image_name, opt, center_thresh=opt.peak_thresh)
        with open(os.path.join(optim_output_folder, image_name.split('/')[-1].split('.')[0] + '.json'),
                  'w') as f_out:
            json.dump(optm_results, f_out, indent=4, sort_keys=True, cls=NumpyEncoder)

    test = pd.read_csv(os.path.join(opt.root_dir, 'sample_submission.csv'))
    for idx, image_id in enumerate(test['ImageId']):
        test['PredictionString'][idx] = predictions[image_id]

    write_to = os.path.join(opt.root_dir, 'submission/{}{}.csv'.format(opt.load_model.split('/')[-2], suffix))
    test.to_csv(write_to, index=False)
    test.head()
    print(write_to)
if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
