from tcar import TestCar
from tcar.utils import LidarPointCloud
import os.path as osp
from nuscenes.utils.data_classes import Box
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from PIL import Image
import cv2
from nuscenes.utils.geometry_utils import view_points, transform_matrix

dataroot = '/home/taewan/TCAR_DATA' 
nusc = TestCar(version='v1.0-trainval', dataroot=dataroot, verbose=True)
cur_scene = nusc.scene[5]

cur_sample = nusc.get('sample', cur_scene['first_sample_token'])
next_sample = nusc.get('sample', cur_sample['next'])

nusc.pointcloud_to_image(cur_sample['next'])
nusc.image_to_pointcloud(cur_sample['next'])