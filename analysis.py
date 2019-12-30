
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
# from keras.preprocessing.image import load_img
from math import sin, cos
from PIL import ImageDraw, Image
import cv2
import os
import json
from collections import namedtuple
import pycocotools.mask as cocoMask
from tqdm import tqdm

class NumpyEncoder(json.JSONEncoder):
    """Helper class to help serialize numpy ndarray"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# convert euler angle to rotation matrix
def euler_to_Rot(yaw, pitch, roll):
    Y = np.array([[cos(yaw), 0, sin(yaw)],
                  [0, 1, 0],
                  [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0],
                  [sin(roll), cos(roll), 0],
                  [0, 0, 1]])
    return np.dot(Y, np.dot(P, R))

def draw_line(image, points):
    color = (255, 0, 0)
    cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, 5)
    cv2.line(image, tuple(points[1][:2]), tuple(points[4][:2]), color, 5)

    cv2.line(image, tuple(points[1][:2]), tuple(points[5][:2]), color, 5)
    cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, 5)
    cv2.line(image, tuple(points[2][:2]), tuple(points[6][:2]), color, 5)
    cv2.line(image, tuple(points[3][:2]), tuple(points[4][:2]), color, 5)
    cv2.line(image, tuple(points[3][:2]), tuple(points[7][:2]), color, 5)

    cv2.line(image, tuple(points[4][:2]), tuple(points[8][:2]), color, 5)
    cv2.line(image, tuple(points[5][:2]), tuple(points[8][:2]), color, 5)

    cv2.line(image, tuple(points[5][:2]), tuple(points[6][:2]), color, 5)
    cv2.line(image, tuple(points[6][:2]), tuple(points[7][:2]), color, 5)
    cv2.line(image, tuple(points[7][:2]), tuple(points[8][:2]), color, 5)
    return image


def draw_points(image, points):
    for idx, (p_x, p_y, p_z) in enumerate(points):
        if idx == 0:
            color = (0, 0, 255)
            size = 1
        else:
            color = (0, 0, 255)
            size = 5
        if idx == 1:
            color = (0, 255, 0)
            size = 5
        if idx == 7:
            color = (255, 0, 0)
            size = 5
        cv2.circle(image, (p_x, p_y), size, color, -1)
    return image

# image coordinate to world coordinate
def img_cor_2_world_cor(img_cor_points, k):
    x_img, y_img, z_img = img_cor_points[0]
    xc, yc, zc = x_img*z_img, y_img*z_img, z_img
    p_cam = np.array([xc, yc, zc])
    xw, yw, zw = np.dot(np.linalg.inv(k), p_cam)
    print(xw, yw, zw)
    print(x, y, z)

def convert_3d_bbox_2d(yaw, pitch, roll, x, y, z, x_l = 1.02, y_l = 0.80, z_l = 2.31):

    # I think the pitch and yaw should be exchanged
    yaw, pitch, roll = -pitch, -yaw, -roll
    Rt = np.eye(4)
    t = np.array([x, y, z])
    Rt[:3, 3] = t
    Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
    Rt = Rt[:3, :]
    P = np.array([[0, 0, 0, 1],
                  [x_l, y_l, -z_l, 1],
                  [x_l, y_l, z_l, 1],
                  [-x_l, y_l, z_l, 1],
                  [-x_l, y_l, -z_l, 1],
                  [x_l, -y_l, -z_l, 1],
                  [x_l, -y_l, z_l, 1],
                  [-x_l, -y_l, z_l, 1],
                  [-x_l, -y_l, -z_l, 1]]).T
    img_cor_points = np.dot(k, np.dot(Rt, P))
    img_cor_points = img_cor_points.T
    img_cor_points[:, 0] /= img_cor_points[:, 2]
    img_cor_points[:, 1] /= img_cor_points[:, 2]

    # call this function before chage the dtype
    # img_cor_2_world_cor(img_cor_points, k)
    img_cor_points = img_cor_points.astype(int)
    return img_cor_points

def convert_3d_ctr_bbox_2d(yaw, pitch, roll, x, y, z, x_l = 1.02, y_l = 0.80, z_l = 2.31):

    # I think the pitch and yaw should be exchanged
    yaw, pitch, roll = -pitch, -yaw, -roll
    Rt = np.eye(4)
    t = np.array([x, y, z])
    Rt[:3, 3] = t
    Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
    Rt = Rt[:3, :]
    P = np.array([[0, 0, 0, 1],
                  [0, y_l, 0, 1],
                  [0, -y_l, 0, 1]]).T

    img_cor_points = np.dot(k, np.dot(Rt, P))
    img_cor_points = img_cor_points.T
    img_cor_points[:, 0] /= img_cor_points[:, 2]
    img_cor_points[:, 1] /= img_cor_points[:, 2]

    # call this function before chage the dtype
    # img_cor_2_world_cor(img_cor_points, k)
    img_cor_points = img_cor_points.astype(int)
    return img_cor_points

def get_theta_ray(cx,
                  intrisics=np.array([[2304.5479, 0,  1686.2379],
           [0, 2305.8757, 1354.9849],
           [0, 0, 1]], dtype=np.float32)):
    """Get thera ray for a specific bbox, emitting from ego car to bbox center

    Example:
        [509.6925, 0.0, 227.006836, 0.0, 509.6925, 118.7186585, 0.0, 0.0, 1.0] * 4


    Args:
        cx: landmark point of bbox used to calculate theta_ray
        intrisics: intrinsics in pixels in orignal scale

    Returns:

    """
    # intrisics = np.array(intrisics).reshape((3, 3))
    px = intrisics[0, 2]
    fx = intrisics[0, 0]
    # py = intrisics[1, 2]
    # fy = intrisics[1, 1]
    dx = cx - px
    theta_ray = np.arctan2(dx, fx)
    return theta_ray

# k is camera instrinsic matrix
k = np.array([[2304.5479, 0,  1686.2379],
           [0, 2305.8757, 1354.9849],
           [0, 0, 1]], dtype=np.float32)


def load_3dlabel(car_model):
    try:
        car_name = car_id2name[int(car_model)][0]
    except:
        car_name = car_model

    json_file = open('/xmotors_ai_shared/datasets/incubator/user/yus/dataset/pku/car_models_json/{}.json'.format(car_name), 'rb')
    data = json.load(json_file)
    return data, car_name

def cvt_2d_bbx(pt):
    minx, miny = np.inf, np.inf
    maxx, maxy = 0, 0
    for idx, (p_x, p_y, p_z) in enumerate(pt):
        minx = min(minx, p_x)
        miny = min(miny, p_y)
        maxx = max(maxx, p_x)
        maxy = max(maxy, p_y)
    return [minx, miny, maxx, maxy]

def fill_hole(image):
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    contour,hier = cv2.findContours(gray,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(gray,[cnt],0,255,-1)
    return gray

# def draw_obj(image, vertices, triangles, color=(0,0,255)):
#     for t in triangles:
#         coord = np.array([vertices[t[0]][:2], vertices[t[1]][:2], vertices[t[2]][:2]], dtype=np.int32)
#         cv2.fillConvexPoly(image, coord, color)
#         cv2.polylines(image, np.int32([coord]), 1, color)
#     image = fill_hole(image)
#     return image

def draw_obj(image, vertices, triangles):
    for t in triangles:
        coord = np.array([vertices[t[0]][:2], vertices[t[1]][:2], vertices[t[2]][:2]], dtype=np.int32)
        cv2.fillConvexPoly(image, coord, 255)
        cv2.polylines(image, np.int32([coord]), 1, 255)
    image = fill_hole(image)
    return image
#         cv2.fillConvexPoly(image, coord, (0,0,255))
#         cv2.polylines(image, np.int32([coord]), 1, (0,0,255))

if __name__ == "__main__":
    # a label and all meta information
    Label = namedtuple('Label', [

        'name',  # The name of a car type
        'id',  # id for specific car type
        'category',  # The name of the car category, 'SUV', 'Sedan' etc
        'categoryId',  # The ID of car category. Used to create ground truth images
        # on category level.
    ])

    models = [
        #     name          id   is_valid  category  categoryId
        Label('baojun-310-2017', 0, '2x', 0),
        Label('biaozhi-3008', 1, '2x', 0),
        Label('biaozhi-liangxiang', 2, '2x', 0),
        Label('bieke-yinglang-XT', 3, '2x', 0),
        Label('biyadi-2x-F0', 4, '2x', 0),
        Label('changanbenben', 5, '2x', 0),
        Label('dongfeng-DS5', 6, '2x', 0),
        Label('feiyate', 7, '2x', 0),
        Label('fengtian-liangxiang', 8, '2x', 0),
        Label('fengtian-MPV', 9, '2x', 0),
        Label('jilixiongmao-2015', 10, '2x', 0),
        Label('lingmu-aotuo-2009', 11, '2x', 0),
        Label('lingmu-swift', 12, '2x', 0),
        Label('lingmu-SX4-2012', 13, '2x', 0),
        Label('sikeda-jingrui', 14, '2x', 0),
        Label('fengtian-weichi-2006', 15, '3x', 1),
        Label('037-CAR02', 16, '3x', 1),
        Label('aodi-a6', 17, '3x', 1),
        Label('baoma-330', 18, '3x', 1),
        Label('baoma-530', 19, '3x', 1),
        Label('baoshijie-paoche', 20, '3x', 1),
        Label('bentian-fengfan', 21, '3x', 1),
        Label('biaozhi-408', 22, '3x', 1),
        Label('biaozhi-508', 23, '3x', 1),
        Label('bieke-kaiyue', 24, '3x', 1),
        Label('fute', 25, '3x', 1),
        Label('haima-3', 26, '3x', 1),
        Label('kaidilake-CTS', 27, '3x', 1),
        Label('leikesasi', 28, '3x', 1),
        Label('mazida-6-2015', 29, '3x', 1),
        Label('MG-GT-2015', 30, '3x', 1),
        Label('oubao', 31, '3x', 1),
        Label('qiya', 32, '3x', 1),
        Label('rongwei-750', 33, '3x', 1),
        Label('supai-2016', 34, '3x', 1),
        Label('xiandai-suonata', 35, '3x', 1),
        Label('yiqi-benteng-b50', 36, '3x', 1),
        Label('bieke', 37, '3x', 1),
        Label('biyadi-F3', 38, '3x', 1),
        Label('biyadi-qin', 39, '3x', 1),
        Label('dazhong', 40, '3x', 1),
        Label('dazhongmaiteng', 41, '3x', 1),
        Label('dihao-EV', 42, '3x', 1),
        Label('dongfeng-xuetielong-C6', 43, '3x', 1),
        Label('dongnan-V3-lingyue-2011', 44, '3x', 1),
        Label('dongfeng-yulong-naruijie', 45, 'SUV', 2),
        Label('019-SUV', 46, 'SUV', 2),
        Label('036-CAR01', 47, 'SUV', 2),
        Label('aodi-Q7-SUV', 48, 'SUV', 2),
        Label('baojun-510', 49, 'SUV', 2),
        Label('baoma-X5', 50, 'SUV', 2),
        Label('baoshijie-kayan', 51, 'SUV', 2),
        Label('beiqi-huansu-H3', 52, 'SUV', 2),
        Label('benchi-GLK-300', 53, 'SUV', 2),
        Label('benchi-ML500', 54, 'SUV', 2),
        Label('fengtian-puladuo-06', 55, 'SUV', 2),
        Label('fengtian-SUV-gai', 56, 'SUV', 2),
        Label('guangqi-chuanqi-GS4-2015', 57, 'SUV', 2),
        Label('jianghuai-ruifeng-S3', 58, 'SUV', 2),
        Label('jili-boyue', 59, 'SUV', 2),
        Label('jipu-3', 60, 'SUV', 2),
        Label('linken-SUV', 61, 'SUV', 2),
        Label('lufeng-X8', 62, 'SUV', 2),
        Label('qirui-ruihu', 63, 'SUV', 2),
        Label('rongwei-RX5', 64, 'SUV', 2),
        Label('sanling-oulande', 65, 'SUV', 2),
        Label('sikeda-SUV', 66, 'SUV', 2),
        Label('Skoda_Fabia-2011', 67, 'SUV', 2),
        Label('xiandai-i25-2016', 68, 'SUV', 2),
        Label('yingfeinidi-qx80', 69, 'SUV', 2),
        Label('yingfeinidi-SUV', 70, 'SUV', 2),
        Label('benchi-SUR', 71, 'SUV', 2),
        Label('biyadi-tang', 72, 'SUV', 2),
        Label('changan-CS35-2012', 73, 'SUV', 2),
        Label('changan-cs5', 74, 'SUV', 2),
        Label('changcheng-H6-2016', 75, 'SUV', 2),
        Label('dazhong-SUV', 76, 'SUV', 2),
        Label('dongfeng-fengguang-S560', 77, 'SUV', 2),
        Label('dongfeng-fengxing-SX6', 78, 'SUV', 2)

    ]

    # name to label object
    car_name2id = {label.name: label for label in models}
    car_id2name = {label.id: label for label in models}

    train = pd.read_csv('/xmotors_ai_shared/datasets/incubator/user/yus/dataset/pku/train.csv')

    plt.close()
    plt.rcParams["axes.grid"] = False
    output_folder = '/xmotors_ai_shared/datasets/incubator/user/yus/dataset/pku/vis_train'

    import glob
    imgs = glob.glob(os.path.join('/xmotors_ai_shared/datasets/incubator/user/yus/dataset/pku/data/images/train_images/', '*jpg'))
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    image = cv2.imread(imgs[0])

    all_r = []
    all_p = []
    all_y = []
    for id in tqdm(range(len(train))):#len(train)):
        gt = {'objects': []}

        img_name = train.loc[id]['ImageId']
        pred_string = train.loc[id]['PredictionString']

        # if not 'ID_6d98a2632' in img_name:
        #     continue

        # image = cv2.imread('/xmotors_ai_shared/datasets/incubator/user/yus/dataset/pku/data/images/train_images/' + img_name + '.jpg')
        # fig, ax = plt.subplots(figsize=(40, 40))
        # img = np.array(image[:,:,::-1])
        items = pred_string.split(' ')
        model_types, yaws, pitches, rolls, xs, ys, zs = [items[i::7] for i in range(7)]
        all_r += rolls
        all_p += pitches
        all_y += yaws

    all_r = np.array(all_r, dtype=np.float)
    all_p = np.array(all_p, dtype=np.float)
    all_y = np.array(all_y, dtype=np.float)

    plt.hist(all_r, normed=True, bins=200)
    plt.show()

    plt.hist(all_p , normed=True, bins=200)
    plt.show()

    plt.hist(all_y, normed=True, bins=200)
    plt.show()

    print(all_y.mean())
    print(all_y.mean())
    print(all_y.mean())
        # for car_model, yaw, pitch, roll, x, y, z in zip(model_types, yaws, pitches, rolls, xs, ys, zs):
        #     overlay = np.zeros((image.shape[0], image.shape[1],3), dtype=np.uint8)
        #     obj = {}
        #     yaw, pitch, roll, x, y, z = [float(x) for x in [yaw, pitch, roll, x, y, z]]


