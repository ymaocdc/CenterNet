from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
import matplotlib.pyplot as plt
import os
from models.decode import ddd_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ddd_post_process
from utils.debugger import Debugger
from utils.ddd_utils import compute_box_3d, project_to_image, alpha2rot_y
from utils.ddd_utils import draw_box_3d, unproject_2d_to_3d

from .base_detector import BaseDetector
try:
  from external.nms import soft_nms,nms
except:
  print('NMS not imported! If you need it,'
        ' do \n cd $CenterNet_ROOT/src/lib/external \n make')

class PkuDetector(BaseDetector):
    def __init__(self, opt):
        super(PkuDetector, self).__init__(opt)
        self.calib = np.array([[2304.5479, 0,  1686.2379, 0],
           [0, 2305.8757, 1354.9849, 0],
           [0, 0, 1, 0]], dtype=np.float32)

    def pre_process(self, image, scale, calib=None):
        height, width = image.shape[0:2]

        inp_height, inp_width = self.opt.input_h, self.opt.input_w
        c = np.array([width / 2, height / 2], dtype=np.float32)
        if self.opt.keep_res:
            s = np.array([inp_width, inp_height], dtype=np.int32)
        else:
            s = np.array([width, height], dtype=np.int32)

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = image  # cv2.resize(image, (width, height))
        inp_image = cv2.warpAffine(
            resized_image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        inp_image = (inp_image.astype(np.float32) / 255.)
        inp_image = (inp_image - self.mean) / self.std
        images = inp_image.transpose(2, 0, 1)[np.newaxis, ...]
        # calib = np.array(calib, dtype=np.float32) if calib is not None \
        #     else self.calib
        calib = self.calib
        images = torch.from_numpy(images)
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio,
                'calib': calib}
        return images, meta

    def process(self, images, return_time=False):
        with torch.no_grad():
            torch.cuda.synchronize()
            output = self.model(images)[-1]
            output['hm'] = output['hm'].sigmoid_()
            output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
            wh = output['wh'] if self.opt.reg_bbox else None
            reg = output['reg'] if self.opt.reg_offset else None

            reg_3d = output['reg_3d_ct'] if self.opt.reg_3d_center else None
            reg_q = output['reg_q'] if self.opt.reg_q else None
            torch.cuda.synchronize()
            forward_time = time.time()

            dets = ddd_decode(output['hm'], output['rot'], output['dep'],
                              output['dim'], wh=wh, reg=reg, K=self.opt.K, reg_3d=reg_3d, q=reg_q)
        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        detections = ddd_post_process(
            dets.copy(), [meta['c']], [meta['s']], [meta['calib']], self.opt)
        self.this_calib = meta['calib']
        return detections[0]

    def merge_outputs(self, detections):
        results = detections[0]
        new_results = []

        for j in range(1, self.num_classes + 1):
            if len(results[j] > 0):
                keep_inds = (results[j][:, 13] > self.opt.peak_thresh)
                results[j] = results[j][keep_inds]

        new_results = np.vstack([results[j] for j in range(1, self.num_classes + 1)])

        keep_inds = nms(new_results, 0.7)
        new_results = new_results[keep_inds]
        new_dict = {}
        for j in range(1, self.num_classes + 1):
            new_dict[j] = []
        new_dict[1] = new_results

        return new_dict

    def debug(self, debugger, images, dets, output, scale=1):
        dets = dets.detach().cpu().numpy()
        img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = ((img * self.std + self.mean) * 255).astype(np.uint8)
        pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hm')
        debugger.add_ct_detection(
            img, dets[0], show_box=self.opt.reg_bbox,
            center_thresh=self.opt.vis_thresh, img_id='det_pred')

    def show_results(self, debugger, image, results, image_or_path_or_tensor):
        debugger.add_3d_detection(
            image, results, self.this_calib, self.opt,
            center_thresh=self.opt.peak_thresh, img_id='add_pred')
        if self.opt.debug > 0:
            fig = plt.figure(figsize=(19, 20))
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            plt.imshow(image[:,:,::-1])

            plt.savefig(os.path.join(self.opt.output_dir, image_or_path_or_tensor.split('/')[-1]), bbox_inches='tight', pad_inches=0, transparent=True)
            # plt.show()

            plt.close(fig)
        debugger.add_bird_view(
            results, self.opt, center_thresh=self.opt.vis_thresh, img_id='bird_pred')
        # debugger.show_all_imgs(pause=self.pause)