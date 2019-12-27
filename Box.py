import numpy as np
from math import sin, cos
import matplotlib.pyplot as plt
import cv2
class Box(object):
    def __init__(self, xmin, ymin, xmax, ymax, bpe_l, bpe_r, fpe_l, fpe_r, K, L, W, H, tx, ty, tz, pitch, cropped, classifier):

        # if not bpe_l is None or not bpe_r is None:
        #     bpe_l, bpe_r = self.fix_bpe(xmin, xmax, bpe_l, bpe_r, cropped)
        # -1: denote known values.
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.bpe_l = bpe_l
        self.bpe_r = bpe_r
        self.fpe_l = fpe_l
        self.fpe_r = fpe_r

        self.K = K
        if type(K) is list:
            self.K = np.array(K).reshape((3, 3))
        self.pitch = pitch
        self.L = L
        self.W = W
        self.H = H
        self.cropped = cropped
        self.valid =self.check_val()
        self.get_direction_visability()

        self.theta_ray = None
        self.local_yaw = None
        self.global_yaw = None
        self.tx, self.ty, self.tz = tx, ty, tz
        self.img_cor_points, self.world_cor_points = None, None
        self.classifier = classifier

    def lift_to_3d(self, mode='optimization'):
        """

        Args:
            mode: can be 'optimization' or 'bpe_similarity'.
                If 'optimiaztion', then 2d/3d constraint is used
                If 'bpe_similarity', then geometric similarity based on 3D width of BPE and its projection
                    will be used to infer distance

        Returns:

        """
        if self.valid.sum() >=2:
            self.theta_ray = self.get_theta_ray(cx=(self.xmin + self.xmax) / 2, intrisics=self.K)


            if self.bpe_l < self.bpe_r:
                self.local_yaw = self.get_local_yaw_when_far(self.xmin, self.xmax, self.bpe_l, self.bpe_r, aspect_ratio=self.L/self.W)

            else:
                self.local_yaw = self.get_local_yaw_when_far(self.xmin, self.xmax, self.fpe_r, self.fpe_l, aspect_ratio=self.L/self.W) + np.pi
            self.local_yaw = np.remainder(self.local_yaw, np.pi*2)
            self.global_yaw = self.get_global_yaw(self.local_yaw, self.theta_ray)

            if mode == 'optimization':
                op_tx, op_ty, op_tz = self.translation_constraints_try_error_yawLWH(True, True, True, True)

            else:
                raise KeyError('mode {} not defined'.format(mode))


            self.img_cor_points, self.world_cor_points = self.map_to_2d(
                op_tx, op_ty, op_tz,
                self.L, self.W, self.H,
                np.pi*2-self.global_yaw, self.pitch, 0)

    def translation_constraints_try_error_yawLWH(self, try_yaw, try_L, try_W, try_H):
        """Improved version with iterative trial and error

        This slows down the code quite a bit. Used for demo now

        TODO: speed up this function
        """
        max_iou = 0
        min_res = np.inf
        ini_global_yaw = self.global_yaw

        if try_yaw:
            # TODO: improve this with binary search
            # if self.direction == 'same':
            #     try_yaw_range = np.arange(-np.pi/2 - np.pi/6, np.pi/2 + np.pi/6, np.pi/60)
            # else:
            #     try_yaw_range = np.arange(np.pi/2  - np.pi / 6, np.pi/2*3 + np.pi / 6, np.pi / 60)
            try_yaw_range = np.arange(ini_global_yaw - np.pi / 12, ini_global_yaw + np.pi / 12, np.pi / 60)
        else:
            try_yaw_range = [ini_global_yaw]

        if try_L:
            try_L_range = np.arange(3.5, 5.5, 0.2)
        else:
            try_L_range = [self.L]

        if try_W:
            try_W_range = np.arange(2,2.3,1)
        else:
             try_W_range = [self.W]

        if try_H:
            # TODO: add car subtype. Go up to 4 only for trucks
            try_H_range = np.arange(1.4, 1.9, 0.1)
        else:
            try_H_range = [self.H]

        for try_global_yaw in try_yaw_range:
            for L in try_L_range:
                for W in try_W_range:
                    for H in try_H_range :
                        # Y = np.array([[np.cos(try_global_yaw), 0, np.sin(try_global_yaw)],
                        #               [0, 1, 0],
                        #               [-np.sin(try_global_yaw), 0, np.cos(try_global_yaw)]])
                        #
                        #
                        # P = np.array([[1, 0, 0],
                        #               [0, cos(self.pitch), -sin(self.pitch)],
                        #               [0, sin(self.pitch), cos(self.pitch)]])
                        R = self.euler_to_Rot(np.pi*2-try_global_yaw, self.pitch, 0).T

                        xmin_candi, xmax_candi, ymin_candi, ymax_candi, \
                        l_bpe_candi, r_bpe_candi, l_fpe_candi, r_fpe_candi = self.box3d_candidate(self.local_yaw, L, W, H)

                        try_xmax = self.xmax
                        try_xmin = self.xmin
                        try_ymax = self.ymax
                        try_ymin = self.ymin
                        try_bpe_l = self.bpe_l
                        try_bpe_r = self.bpe_r
                        try_fpe_l = self.fpe_l
                        try_fpe_r = self.fpe_r

                        num_constrain = 4
                        bbox = [try_xmin, try_ymin, try_xmax, try_ymax]
                        X = np.bmat([xmin_candi, ymin_candi,
                                     xmax_candi, ymax_candi])

                        if self.vis == 0:
                            bbox.append(try_bpe_l)
                            X = np.concatenate((X, np.bmat(l_bpe_candi)), axis=1)
                            num_constrain += 1
                        if self.vis ==1:
                            bbox.append(try_bpe_r)
                            X = np.concatenate((X, np.bmat(r_bpe_candi)), axis=1)
                            num_constrain += 1
                        if self.vis == 2:
                            bbox.append(try_fpe_l)
                            X = np.concatenate((X, np.bmat(l_fpe_candi)), axis=1)
                            num_constrain += 1
                        if self.vis == 3:
                            bbox.append(try_fpe_r)
                            X = np.concatenate((X, np.bmat(r_fpe_candi)), axis=1)
                            num_constrain += 1

                        A = np.zeros((num_constrain, 3))
                        b = np.zeros((num_constrain, 1))
                        I = np.identity(3)

                        # X: [x, y, z] in object coordinate
                        X = X.reshape(num_constrain, 3).T

                        # construct equation (4, 3)
                        for i in range(num_constrain):
                            matrice = np.bmat([[I, np.matmul(R, X[:, i])]])
                            M = np.matmul(self.K, matrice)

                            if i in [0, 2, 4]:
                                A[i, :] = M[0, 0:3] - bbox[i] * M[2, 0:3]
                                b[i, :] = M[2, 3] * bbox[i] - M[0, 3]
                            else:
                                A[i, :] = M[1, 0:3] - bbox[i] * M[2, 0:3]
                                b[i, :] = M[2, 3] * bbox[i] - M[1, 3]
                        # solve x, y, z, using method of least square
                        Tran = np.matmul(np.linalg.pinv(A), b)
                        res = np.matmul(A, Tran) - b

                        tx, ty, tz = [float(np.around(tran, 2)) for tran in Tran]
                        img_cor_points, world_cor_points = self.map_to_2d(tx, ty, tz, L, W, H, np.pi*2-try_global_yaw, self.pitch, 0)

                        # proj_bbox = [min(img_cor_points[:,0]), min(img_cor_points[:,1]), max(img_cor_points[:,0]), max(img_cor_points[:,1])]
                        # iou = self.get_iou(bbox, proj_bbox)

                        # if iou > max_iou and np.linalg.norm(res) < min_res:
                        if np.linalg.norm(res) < min_res:
                            self.global_yaw = try_global_yaw
                            best_tx, best_ty, best_tz = tx, ty, tz
                            self.L, self.W, self.H = L, W, H
                            # max_iou = iou
                            min_res = np.linalg.norm(res)

        # img_cor_points, world_cor_points = self.map_to_2d(self.tx, self.ty, self.tz, self.L, self.W, self.H, -self.global_yaw,
        #                                                   self.pitch, 0)
        # img = np.zeros((2710, 3384, 3), dtype = np.uint8)
        # img = self.draw_corners(img, img_cor_points)
        # img = self.draw_lines(img, img_cor_points, style='fb')
        # plt.clf()
        # plt.imshow(img[:, :, ::-1])
        # plt.show()
        return best_tx, best_ty, best_tz

    @staticmethod
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

    def map_to_2d(self, X ,Y, Z, L, W, H, yaw, pitch, roll, offset_from_center=None):
        Rt = np.eye(4)
        t = np.array([X ,Y, Z])
        Rt[:3, 3] = t
        Rt[:3, :3] = self.euler_to_Rot(yaw, pitch, roll).T
        Rt = Rt[:3, :]
        x_l, y_l, z_l = W / 2, H / 2, L / 2
        P = np.array([[0, 0, 0, 1],
                      [x_l, y_l, -z_l, 1],
                      [x_l, y_l, z_l, 1],
                      [-x_l, y_l, z_l, 1],
                      [-x_l, y_l, -z_l, 1],
                      [x_l, -y_l, -z_l, 1],
                      [x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, -z_l, 1]]).T
        world_cor_points = np.dot(Rt, P)
        img_cor_points = np.dot(self.K, np.dot(Rt, P))
        img_cor_points = img_cor_points.T
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]
        img_cor_points = img_cor_points.astype(int)
        return img_cor_points, world_cor_points


    @staticmethod
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
        intrisics = np.array(intrisics).reshape((3, 3))
        px = intrisics[0, 2]
        fx = intrisics[0, 0]
        # py = intrisics[1, 2]
        # fy = intrisics[1, 1]
        dx = cx - px
        theta_ray = np.arctan2(dx, fx)
        return theta_ray

    @staticmethod
    def get_global_yaw(local_yaw, theta_ray):
        """Convert from local yaw to global yaw"""
        return theta_ray + local_yaw

    @staticmethod
    def get_alpha(xmin, xmax, bpe_l, bpe_r):
        bpe_l = max(bpe_l, xmin)
        bpe_r = min(bpe_r, xmax)
        width = xmax - xmin
        bpe_width = bpe_r - bpe_l
        margin_right = xmax - bpe_r
        margin_left = bpe_l - xmin
        assert width > 0 and bpe_width > 0 and margin_left >= 0 and margin_right >= 0
        alpha = bpe_width / width
        heading_right = margin_right > margin_left
        return alpha, heading_right

    @staticmethod
    def get_local_yaw_when_far(xmin, xmax, bpe_l, bpe_r, aspect_ratio=2.6):
        """ Calculate local yaw in rad based on the approximation equation (when distance D >> aspect ratio L)
            Local yaw = 0 when car facing forward, +pi/2 when facing left, -pi/2 when facing right
            Backplane ratio alpha = backplane width/bbox width
        """
        alpha, heading_right = Box.get_alpha(xmin, xmax, bpe_l, bpe_r)
        local_yaw_predict = np.arctan((1 - alpha) / (aspect_ratio * alpha))
        # make local_yaw negative when facing right
        # if heading_right:
        #     local_yaw_predict = -local_yaw_predict
        if not heading_right:
            local_yaw_predict = np.pi * 2 - local_yaw_predict
        return local_yaw_predict

    @staticmethod
    def box3d_candidate(local_yaw, L=4.6, W=2, H=1.6, soft_range=0):
        """
        give local yaw and 3d real size of object, find the corresponding 3d Points that define the 2d bbox.
        """
        x_l = W / 2
        y_l = H / 2
        z_l = L / 2

        # K[[I 0^T]T | [[RX_3d 1]T,[T 1]T] = X_2d
        X3d = np.array([[-x_l, -y_l, -z_l],
                        [-x_l, -y_l, z_l],
                        [x_l, -y_l, z_l],
                        [x_l, -y_l, -z_l],
                        [-x_l, y_l, -z_l],
                        [-x_l, y_l, z_l],
                        [x_l, y_l, z_l],
                        [x_l, y_l, -z_l]])
        point1 = X3d[0, :]
        point2 = X3d[1, :]
        point3 = X3d[2, :]
        point4 = X3d[3, :]
        point5 = X3d[4, :]
        point6 = X3d[5, :]
        point7 = X3d[6, :]
        point8 = X3d[7, :]

        # set up projection relation based on local orientation
        xmin_candi = xmax_candi = ymin_candi = ymax_candi = l_bpe_candi = r_bpe_candi = l_fpe_candi = r_fpe_candi = None

        if 0 <= local_yaw < np.pi / 2:
            xmin_candi = point5
            xmax_candi = point7
            ymin_candi = point2
            ymax_candi = point8
            l_bpe_candi = point1
            r_bpe_candi = point4
            l_fpe_candi = point6
            r_fpe_candi = point3

        # note that here if facing towards ego car, bpe is actually fpe
        if np.pi / 2 <= local_yaw <= np.pi:
            xmin_candi = point4
            xmax_candi = point6
            ymin_candi = point1
            ymax_candi = point7
            l_bpe_candi = point5
            r_bpe_candi = point8
            l_fpe_candi = point2
            r_fpe_candi = point3

        # note that here if facing towards ego car, bpe is actually fpe
        if np.pi < local_yaw <= 3 / 2 * np.pi:
            xmin_candi = point3
            xmax_candi = point5
            ymin_candi = point4
            ymax_candi = point6
            l_bpe_candi = point1
            r_bpe_candi = point8
            l_fpe_candi = point2
            r_fpe_candi = point7

        if 3 * np.pi / 2 < local_yaw <= 2 * np.pi:
            xmin_candi = point2
            xmax_candi = point4
            ymin_candi = point3
            ymax_candi = point5
            l_bpe_candi = point1
            r_bpe_candi = point8
            l_fpe_candi = point6
            r_fpe_candi = point7

        div = soft_range * np.pi / 180
        if 0 < local_yaw < div or 2 * np.pi - div < local_yaw < 2 * np.pi:
            xmin_candi = point1
            xmax_candi = point8
            ymin_candi = point4
            ymax_candi = point5
            l_bpe_candi = None
            r_bpe_candi = None

        if np.pi - div < local_yaw < np.pi + div:
            xmin_candi = point3
            xmax_candi = point6
            ymin_candi = point2
            ymax_candi = point7
            l_bpe_candi = None
            r_bpe_candi = None

        return xmin_candi, xmax_candi, ymin_candi, ymax_candi, l_bpe_candi, r_bpe_candi, l_fpe_candi, r_fpe_candi




    def get_iou(self, bbox, proj_bbox):
        intersect_mins = np.maximum(proj_bbox[0:2], bbox[0:2])
        intersect_maxes = np.minimum(proj_bbox[2:4], bbox[2:4])
        intersect_wh = np.clip(intersect_maxes - intersect_mins, 0, intersect_maxes - intersect_mins)
        intersect_areas = intersect_wh[0] * intersect_wh[1]

        boxes_1_areas = (proj_bbox[2] - proj_bbox[0]) * (proj_bbox[3] - proj_bbox[1])
        boxes_2_areas = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        union_areas = boxes_1_areas + boxes_2_areas - intersect_areas
        iou = intersect_areas / union_areas
        return iou
    def get_direction_visability(self):
        self.vis = None
        if self.bpe_l < self.bpe_r:
            self.direction = 'same'
        else:
            self.direction = 'oppo'

        if self.direction == 'same':
            if self.valid[0] and self.valid[1]:#two bpe exists
                if abs(self.bpe_l-self.xmin) < abs(self.bpe_r - self.xmax):
                    self.bpe_l = self.xmin
                    self.vis = 1
                else:
                    self.bpe_r = self.xmax
                    self.vis = 0
            elif self.valid[0]:
                self.vis = 0
            elif self.valid[1]:
                self.vis = 1
            else:
                if self.bpe_l > self.xmax:
                    self.xmax = self.bpe_r
                    self.vis = 3
                elif self.bpe_r < self.xmin:
                    self.xmin = self.bpe_l
                    self.vis = 2
        else:
            # two fpe exists, opposite direction use fpe first
            if self.valid[2] and self.valid[3]:
                if abs(self.fpe_r-self.xmin) > abs(self.fpe_l-self.xmax):
                    self.fpe_l = self.xmax
                    self.vis = 3
                else:
                    self.fpe_r = self.xmin
                    self.vis = 2
            elif self.valid[2]:
                self.vis = 2
            elif self.valid[3]:
                self.vis = 3
            else:
                if self.fpe_l < self.xmin:
                    self.xmin = self.fpe_r
                    self.vis = 1
                elif self.fpe_r > self.xmax:
                    self.xmax = self.fpe_l
                    self.vis = 0
        # if self.bpe_l < self.bpe_r:
        #     self.direction = 'same'
        #     if self.valid[0] and self.valid[1]:#two bpe exists
        #         if abs(self.bpe_l-self.xmin) < abs(self.bpe_r - self.xmax):
        #             self.bpe_l = self.xmin
        #             self.vis = 1
        #         else:
        #             self.bpe_r = self.xmax
        #             self.vis = 0
        #     elif self.valid[0]:
        #         self.vis = 0
        #     elif self.valid[1]:
        #         self.vis = 1
        # else:
        #     self.direction = 'oppo'
        #     if self.valid[2] and self.valid[3]:
        #         if abs(self.fpe_r-self.xmin) > abs(self.fpe_l-self.xmax):
        #             self.fpe_l = self.xmax
        #             self.vis = 3
        #         else:
        #             self.fpe_r = self.xmin
        #             self.vis = 2
        #     elif self.valid[2]:
        #         self.vis = 2
        #     elif self.valid[3]:
        #         self.vis = 3


    def check_val(self):
        valid = []
        if self.bpe_l > 0 and self.bpe_l < 3384:
            valid.append(True)
        else:
            valid.append(False)
        if self.bpe_r > 0 and self.bpe_r < 3384:
            valid.append(True)
        else:
            valid.append(False)
        if self.fpe_l > 0 and self.fpe_l < 3384:
            valid.append(True)
        else:
            valid.append(False)
        if self.fpe_r > 0 and self.fpe_r < 3384:
            valid.append(True)
        else:
            valid.append(False)
        return np.array(valid)

    def draw_lines(self,img, points, linewidth=3, style='allw'):

        if style == 'fb':  # front back
            color = (0, 255, 255)
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
        elif style == 'tb':  # top bottom
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
        elif style == 'allw':  # front back
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

    def draw_corners(self, img, points):
        "Draw 8 corners and centroid of 3d bbox on image"
        for idx, (p_x, p_y, p_z) in enumerate(points):
            if idx == 0:
                color = (0, 128, 255)
            elif idx == 1:
                color = (0, 255, 255)
            elif idx == 7:
                color = (255, 0, 0)
            else:
                color = (255, 255, 255)
            cv2.circle(img, (p_x, p_y), 20, color, -1)
        return img
