import os, glob, json
import numpy as np
import cv2
import matplotlib.pyplot as plt

from Box import Box
from tqdm import tqdm

calib = np.array([[2304.5479, 0,  1686.2379],
           [0, 2305.8757, 1354.9849],
           [0, 0, 1]], dtype=np.float32)

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
        cv2.circle(img, (p_x, p_y), 3, color, -1)
    return img

def draw_bpe(img, bpe_l, bpe_r, ymin, ymax, color=(0,255,255), linewidth=2):
    cv2.line(img, (bpe_l, ymin), (bpe_l, ymax), color, linewidth)
    cv2.line(img, (bpe_r, ymin), (bpe_r, ymax), color, linewidth)
    return img
def toint(x):
    return [int(i) for i in x]

resutls_dir = '/xmotors_ai_shared/datasets/incubator/user/yus/dataset/pku/resutls'

files = glob.glob(os.path.join(resutls_dir, '*json'))
print('number of files:', len(files))

for file in tqdm(files[9:]):
    with open(file, 'r') as fin:
        res = json.load(fin)
    file = res['file']
    org_im = cv2.imread(file)
    img = org_im.copy()
    for obj in res['objects']:
        im = org_im.copy()
        xmin, ymin, xmax, ymax = toint(obj['bbox'])
        bpe_l, bpe_r = toint(obj['BPE'])
        fpe_l, fpe_r = toint(obj['FPE'])
        H, W, L = obj['dim']
        tx, ty, tz = obj['loc']
        pitch = obj['pitch']


        cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (255,255,255), 5)

        cv2.line(im, (int(bpe_l), ymin),
                 (int(bpe_l), ymax), (0, 0, 255), 3,
                 lineType=cv2.LINE_AA)
        cv2.line(im, (int(bpe_r), ymin),
                 (int(bpe_r), ymax), (255, 0, 0), 3,
                 lineType=cv2.LINE_AA)

        cv2.line(im, (int(fpe_l), ymin),
                 (int(fpe_l), ymax), (0, 255, 255), 3,
                 lineType=cv2.LINE_AA)
        cv2.line(im, (int(fpe_r), ymin),
                 (int(fpe_r), ymax), (0, 255, 0), 3,
                 lineType=cv2.LINE_AA)

        # fig = plt.figure(figsize=(10,10))
        # plt.imshow(im[:,:,::-1])
        # plt.show()
        # plt.close(fig)
        box3d = Box(xmin, ymin, xmax, ymax, bpe_l, bpe_r, fpe_l, fpe_r, calib, L, W, H, tx, ty, tz, pitch, None, None)
        box3d.lift_to_3d()


        img_cor_points = box3d.img_cor_points
        if not img_cor_points is None:
            img_cor_points = img_cor_points
            img = draw_corners(img, img_cor_points)
            img = draw_lines(img, img_cor_points, style='fb')
            text = '{} {} {}'.format(np.round(box3d.tx, 1), np.round(box3d.ty, 1), np.round(box3d.tz, 1))
            cv2.putText(img, text, tuple((img_cor_points[1][0] - 5, img_cor_points[1][1] - 20)), 1, 0.5,
                        (0, 50, 255), 1, cv2.LINE_AA)
            text = '{} {} {}'.format(np.round(box3d.L, 1),
                                     np.round(box3d.W, 1), np.round(box3d.H, 1))
            cv2.putText(img, text, tuple(img_cor_points[1][:2] - 5), 1, 0.5, (0, 50, 255), 1, cv2.LINE_AA)
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(img[:, :, ::-1])
    plt.show()
    plt.close(fig)