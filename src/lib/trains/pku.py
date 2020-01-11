from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss, L1Loss, BinRotLoss
from models.decode import ddd_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ddd_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer


class DddLoss(torch.nn.Module):
    def __init__(self, opt):
        super(DddLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = L1Loss()
        self.crit_rot = BinRotLoss()
        if opt.reg_pitch:
            self.crit_pitch = BinRotLoss()
        self.opt = opt

    def forward(self, outputs, batch):
        opt = self.opt

        hm_loss, dep_loss, rot_loss, dim_loss, pitch_loss, reg_3d_center_loss, reg_BPE_loss, reg_FPE_loss = 0, 0, 0, 0, 0, 0, 0, 0
        wh_loss, off_loss, roll_loss, req_q_loss = 0, 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            output['hm'] = _sigmoid(output['hm'])
            output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.

            if opt.eval_oracle_dep:
                output['dep'] = torch.from_numpy(gen_oracle_map(
                    batch['dep'].detach().cpu().numpy(),
                    batch['ind'].detach().cpu().numpy(),
                    opt.output_w, opt.output_h)).to(opt.device)

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.dep_weight >= 0:
                dep_loss += self.crit_reg(output['dep'], batch['rot_mask'],
                                          batch['ind'], batch['dep']) / opt.num_stacks
            if opt.dim_weight >= 0:
                dim_loss += self.crit_reg(output['dim'], batch['reg_mask'],
                                          batch['ind'], batch['dim']) / opt.num_stacks
            if opt.rot_weight >=0:
                rot_loss += self.crit_rot(output['rot'], batch['rot_mask'],
                                          batch['ind'], batch['rotbin'],
                                          batch['rotres']) / opt.num_stacks
            if opt.pitch_weight >= 0 and opt.reg_pitch:
                pitch_loss += self.crit_reg(output['reg_pitch'], batch['reg_pitch_mask'],
                                          batch['ind'], batch['reg_pitch']) / opt.num_stacks
            if opt.roll_weight >= 0 and opt.reg_roll:
                roll_loss += self.crit_reg(output['reg_roll'], batch['reg_roll_mask'],
                                          batch['ind'], batch['reg_roll']) / opt.num_stacks
            if opt.reg_bbox and opt.wh_weight >= 0:
                wh_loss += self.crit_reg(output['wh'], batch['rot_mask'],
                                         batch['ind'], batch['wh']) / opt.num_stacks
            if opt.reg_offset and opt.off_weight >= 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks
            if opt.reg_3d_center and opt.reg_3d_center_weight >= 0:
                reg_3d_center_loss += self.crit_reg(output['reg_3d_ct'], batch['reg_3d_ct_mask'],
                                          batch['ind'], batch['reg_3d_ct']) / opt.num_stacks
            if opt.reg_BPE and opt.reg_BPE_weight >=0:
                reg_BPE_loss += self.crit_reg(output['reg_BPE'], batch['reg_BPE_mask'],
                                                    batch['ind'], batch['reg_BPE']) / opt.num_stacks
            if opt.reg_FPE and opt.reg_FPE_weight >= 0:
                reg_FPE_loss += self.crit_reg(output['reg_FPE'], batch['reg_FPE_mask'],
                                              batch['ind'], batch['reg_FPE']) / opt.num_stacks
            if opt.reg_q and opt.q_weight >= 0:
                req_q_loss += self.crit_reg(output['reg_q'], batch['rot_mask'],
                                          batch['ind'], batch['reg_q']) / opt.num_stacks

        loss = opt.hm_weight * hm_loss + opt.dep_weight * dep_loss + \
               opt.dim_weight * dim_loss + opt.rot_weight * rot_loss + \
               opt.wh_weight * wh_loss + opt.off_weight * off_loss

        loss_stats = {'hm_loss': hm_loss, 'dep_loss': dep_loss,
                      'dim_loss': dim_loss, 'rot_loss': rot_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss}

        if self.opt.reg_pitch:
            loss += opt.pitch_weight * pitch_loss
            loss_stats.update({'reg_pitch_loss': pitch_loss})
        if self.opt.reg_roll:
            loss += opt.reg_roll_weight * roll_loss
            loss_stats.update({'reg_roll_loss': roll_loss})
        if self.opt.reg_3d_center:
            loss += opt.reg_3d_center_weight*reg_3d_center_loss
            loss_stats.update({'reg_3d_center_loss': reg_3d_center_loss})
        if self.opt.reg_BPE:
            loss += opt.reg_BPE_weight * reg_BPE_loss
            loss_stats.update({'reg_BPE_loss': reg_BPE_loss})
        if self.opt.reg_FPE:
            loss += opt.reg_FPE_weight * reg_FPE_loss
            loss_stats.update({'reg_FPE_loss': reg_FPE_loss})
        if self.opt.reg_q:
            loss += req_q_loss*opt.q_weight
            loss_stats.update({'reg_q_loss': req_q_loss})
        loss_stats.update({'loss': loss})

        return loss, loss_stats


class PkuTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(PkuTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'dep_loss', 'dim_loss', 'rot_loss',
                       'wh_loss', 'off_loss']
        if self.opt.reg_pitch:
            loss_states = loss_states + ['reg_pitch_loss']
        if self.opt.reg_3d_center:
            loss_states = loss_states + ['reg_3d_center_loss']
        if self.opt.reg_FPE:
            loss_states = loss_states + ['reg_FPE_loss']
        if self.opt.reg_BPE:
            loss_states = loss_states + ['reg_BPE_loss']
        if self.opt.reg_roll:
            loss_states = loss_states + ['reg_roll_loss']
        if self.opt.reg_q:
            loss_states = loss_states + ['reg_q_loss']
        loss = DddLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id):
        opt = self.opt
        wh = output['wh'] if opt.reg_bbox else None
        reg = output['reg'] if opt.reg_offset else None
        dets = ddd_decode(output['hm'], output['rot'], output['dep'],
                          output['dim'], wh=wh, reg=reg, K=opt.K)

        # x, y, score, r1-r8, depth, dim1-dim3, cls
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        calib = batch['meta']['calib'].detach().numpy()
        # x, y, score, rot, depth, dim1, dim2, dim3
        # if opt.dataset == 'gta':
        #   dets[:, 12:15] /= 3
        dets_pred = ddd_post_process(
            dets.copy(), batch['meta']['c'].detach().numpy(),
            batch['meta']['s'].detach().numpy(), calib, opt)
        dets_gt = ddd_post_process(
            batch['meta']['gt_det'].detach().numpy().copy(),
            batch['meta']['c'].detach().numpy(),
            batch['meta']['s'].detach().numpy(), calib, opt)
        # for i in range(input.size(0)):
        for i in range(1):
            debugger = Debugger(dataset=opt.dataset, ipynb=(opt.debug == 3),
                                theme=opt.debugger_theme)
            img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img * self.opt.std + self.opt.mean) * 255.).astype(np.uint8)
            pred = debugger.gen_colormap(
                output['hm'][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'hm_pred')
            debugger.add_blend_img(img, gt, 'hm_gt')
            # decode
            debugger.add_ct_detection(
                img, dets[i], show_box=opt.reg_bbox, center_thresh=opt.center_thresh,
                img_id='det_pred')
            debugger.add_ct_detection(
                img, batch['meta']['gt_det'][i].cpu().numpy().copy(),
                show_box=opt.reg_bbox, img_id='det_gt')
            debugger.add_3d_detection(
                batch['meta']['image_path'][i], dets_pred[i], calib[i],
                center_thresh=opt.center_thresh, img_id='add_pred')
            debugger.add_3d_detection(
                batch['meta']['image_path'][i], dets_gt[i], calib[i],
                center_thresh=opt.center_thresh, img_id='add_gt')
            # debugger.add_bird_view(
            #   dets_pred[i], center_thresh=opt.center_thresh, img_id='bird_pred')
            # debugger.add_bird_view(dets_gt[i], img_id='bird_gt')
            debugger.add_bird_views(
                dets_pred[i], dets_gt[i],
                center_thresh=opt.center_thresh, img_id='bird_pred_gt')

            # debugger.add_blend_img(img, pred, 'out', white=True)
            debugger.compose_vis_add(
                batch['meta']['image_path'][i], dets_pred[i], calib[i],
                opt.center_thresh, pred, 'bird_pred_gt', img_id='out')
            # debugger.add_img(img, img_id='out')
            if opt.debug == 4:
                debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
            else:
                debugger.show_all_imgs(pause=True)

    def save_result(self, output, batch, results):
        opt = self.opt
        wh = output['wh'] if opt.reg_bbox else None
        reg = output['reg'] if opt.reg_offset else None
        dets = ddd_decode(output['hm'], output['rot'], output['dep'],
                          output['dim'], wh=wh, reg=reg, K=opt.K)

        # x, y, score, r1-r8, depth, dim1-dim3, cls
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        calib = batch['meta']['calib'].detach().numpy()
        # x, y, score, rot, depth, dim1, dim2, dim3
        dets_pred = ddd_post_process(
            dets.copy(), batch['meta']['c'].detach().numpy(),
            batch['meta']['s'].detach().numpy(), calib, opt)
        img_id = batch['meta']['img_id'].detach().numpy()[0]
        results[img_id] = dets_pred[0]
        for j in range(1, opt.num_classes + 1):
            keep_inds = (results[img_id][j][:, -1] > opt.center_thresh)
            results[img_id][j] = results[img_id][j][keep_inds]