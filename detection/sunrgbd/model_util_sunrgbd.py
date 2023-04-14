# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))


class SunrgbdDatasetConfig(object):
    def __init__(self):
        self.num_class = 10
        self.num_heading_bin = 12
        self.num_size_cluster = self.num_class

        self.type2class = {'bed': 0, 'table': 1, 'sofa': 2, 'chair': 3, 'toilet': 4, 'desk': 5, 'dresser': 6,
                           'night_stand': 7, 'bookshelf': 8, 'bathtub': 9}
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        self.type2onehotclass = {'bed': 0, 'table': 1, 'sofa': 2, 'chair': 3, 'toilet': 4, 'desk': 5, 'dresser': 6,
                                 'night_stand': 7, 'bookshelf': 8, 'bathtub': 9}
        self.type_mean_size = {'bathtub': np.array([0.765840, 1.398258, 0.472728]),
                               'bed': np.array([2.114256, 1.620300, 0.927272]),
                               'bookshelf': np.array([0.404671, 1.071108, 1.688889]),
                               'chair': np.array([0.591958, 0.552978, 0.827272]),
                               'desk': np.array([0.695190, 1.346299, 0.736364]),
                               'dresser': np.array([0.528526, 1.002642, 1.172878]),
                               'night_stand': np.array([0.500618, 0.632163, 0.683424]),
                               'sofa': np.array([0.923508, 1.867419, 0.845495]),
                               'table': np.array([0.791118, 1.279516, 0.718182]),
                               'toilet': np.array([0.699104, 0.454178, 0.756250])}
        self.mean_size_arr = np.zeros((self.num_size_cluster, 3))
        for i in range(self.num_size_cluster):
            self.mean_size_arr[i, :] = self.type_mean_size[self.class2type[i]]

    def size2class(self, size, type_name):
        ''' Convert 3D box size (l,w,h) to size class and size residual '''
        size_class = self.type2class[type_name]
        size_residual = size - self.type_mean_size[type_name]
        return size_class, size_residual

    def class2size(self, pred_cls, residual):
        ''' Inverse function to size2class '''
        mean_size = self.type_mean_size[self.class2type[pred_cls]]
        return mean_size + residual

    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class
            [optinal] also small regression number from  
            class center angle to current angle.
           
            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle
        '''
        num_class = self.num_heading_bin
        angle = angle % (2 * np.pi)
        assert (angle >= 0 and angle <= 2 * np.pi)
        angle_per_class = 2 * np.pi / float(num_class)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
        return class_id, residual_angle

    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class '''
        num_class = self.num_heading_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle > np.pi:
            angle = angle - 2 * np.pi
        return angle

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle * -1
        return obb

class SunrgbdDatasetConfig_Uncommon(object):
    def __init__(self):
        self.num_class = 21
        self.num_heading_bin = 12
        self.num_size_cluster = self.num_class

        self.type2class = {'bag': 0, 'blinds': 1,'books': 2, 'box': 3,
                               'cabinet': 4, 'clothes': 5, 'counter': 6, 'curtain': 7,
                               'door': 8, 'fridge': 9, 'lamp': 10,
                               'mirror': 11, 'paper': 12, 'person': 13,
                               'picture': 14, 'pillow': 15, 'shower_curtain': 16,
                               'sink': 17, 'towel': 18, 'tv': 19, 'whiteboard': 20, 'window': 21}
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        self.type2onehotclass = self.type2class
        self.type_mean_size = {'bag': np.array([0.355188, 0.378856, 0.346023]),
                               'blinds': np.array([0.151844, 0.967588, 1.358532]),
                               'books': np.array([0.287604, 0.295896, 0.205587]),
                               'box': np.array([0.351994, 0.364480, 0.309092]),
                               'cabinet': np.array([0.571631, 1.214407, 0.963636]),
                               'clothes': np.array([0.378322, 0.453536, 0.345454]),
                               'counter': np.array([0.760644, 2.236330, 0.850000]),
                               'curtain': np.array([0.257625, 0.711000, 1.659091]),
                               'door': np.array([0.160932, 0.690090, 1.880588]),
                               'fridge': np.array([0.732086, 0.754600, 1.650000]),
                               'lamp': np.array([0.367022, 0.379614, 0.690910]),
                               'mirror': np.array([0.140872, 0.706094, 0.990909]),
                               'paper': np.array([0.238536, 0.245622, 0.090910]),
                               'person': np.array([0.551934, 0.630834, 1.218182]),
                               'picture': np.array([0.118182, 0.455344, 0.472728]),
                               'pillow': np.array([0.355497, 0.560770, 0.318182]),
                               'shower_curtain': np.array([0.243048, 0.415382, 1.154546]),
                               'sink': np.array([0.502248, 0.599351, 0.457344]),
                               'towel': np.array([0.197596, 0.313472, 0.350000]),
                               'tv': np.array([0.248484, 0.800022, 0.608334]),
                               'whiteboard': np.array([0.140555, 1.654753, 1.045454]),
                               'window': np.array([0.138318, 1.854300, 1.009090])}
        self.mean_size_arr = np.zeros((self.num_size_cluster, 3))
        for i in range(self.num_size_cluster):
            self.mean_size_arr[i, :] = self.type_mean_size[self.class2type[i]]

    def size2class(self, size, type_name):
        ''' Convert 3D box size (l,w,h) to size class and size residual '''
        size_class = self.type2class[type_name]
        size_residual = size - self.type_mean_size[type_name]
        return size_class, size_residual

    def class2size(self, pred_cls, residual):
        ''' Inverse function to size2class '''
        mean_size = self.type_mean_size[self.class2type[pred_cls]]
        return mean_size + residual

    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class
            [optinal] also small regression number from
            class center angle to current angle.

            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle
        '''
        num_class = self.num_heading_bin
        angle = angle % (2 * np.pi)
        assert (angle >= 0 and angle <= 2 * np.pi)
        angle_per_class = 2 * np.pi / float(num_class)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
        return class_id, residual_angle

    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class '''
        num_class = self.num_heading_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle > np.pi:
            angle = angle - 2 * np.pi
        return angle

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle * -1
        return obb

