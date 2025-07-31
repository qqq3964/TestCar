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

dataroot = '/home/taewan/TCAR_DATA_origin' 
nusc = TestCar(version='v1.0-trainval', dataroot=dataroot, verbose=True)

# cur_sample = nusc.get('sample', cur_scene['first_sample_token'])
# next_sample = nusc.get('sample', cur_sample['next'])

def image_to_pointcloud(sample_token: str,
                        min_dist: float = 0.0):
    camera_sensors = {
        'CAM_FRONT',
        'CAM_FRONT_LEFT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT'
    }
    sensor = 'LIDAR_TOP'
    
    cur_sample = nusc.get('sample', sample_token)
    
    total_lidar_points = list()
    total_lidar_colors = list()
    for camera_sensor in camera_sensors:
        # camera
        cam_rec = nusc.get('sample_data', cur_sample['data'][camera_sensor])
        cs_rec = nusc.get('calibrated_sensor', cam_rec['calibrated_sensor_token'])
        ego2cam = transform_matrix(cs_rec['translation'], Quaternion(cs_rec['rotation']), inverse=False)
            
        dist_intrinsic = np.array(cs_rec['camera_intrinsic'])
        distortion = np.array(cs_rec['distortion'])
        
        # lidar
        lidar_rec = nusc.get('sample_data', cur_sample['data'][sensor])
        cs_rec = nusc.get('calibrated_sensor', lidar_rec['calibrated_sensor_token'])
        lidar2ego = transform_matrix(cs_rec['translation'], Quaternion(cs_rec['rotation']), inverse=True) # inverse false is ego2lidar
        
        # sensor
        img_path = osp.join(dataroot, cam_rec['filename'])
        pcl_path = osp.join(dataroot, lidar_rec['filename'])

        img = np.array(Image.open(img_path))
        lidar = LidarPointCloud.from_file(pcl_path)
        
        h, w = img.shape[:2]

        if (camera_sensor != 'CAM_FRONT'):
            balance = 0.0  # 0: maximum crop, 1: maximum FOV
            intrinsic = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                dist_intrinsic, distortion, (w, h), np.eye(3), balance=balance
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                dist_intrinsic, distortion, np.eye(3), intrinsic, (w, h), cv2.CV_16SC2
            )
            img = cv2.remap(img, map1, map2,
                                    interpolation=cv2.INTER_LINEAR)
        else:
            img = cv2.undistort(img, dist_intrinsic, distortion)
            intrinsic = dist_intrinsic
        
        lidar.transform(lidar2ego)
        lidar.transform(ego2cam)
        
        depths = lidar.points[2, :]
        normalized_points = lidar.points[:3, :] / depths
        points = intrinsic @ normalized_points
        
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < w - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < h - 1)
        
        points = points[:, mask]
        coloring = depths[mask]
        
        normalized_points = np.linalg.inv(intrinsic) @ points
        camera_points = normalized_points * coloring
        homo = np.ones((1, camera_points.shape[1])) 
        points_homo = np.vstack((camera_points, homo))
        points_homo = np.linalg.inv(ego2cam) @ points_homo
        lidar_points = np.linalg.inv(lidar2ego) @ points_homo
        
        # rgb matched lidar points
        rows = np.round(points[1, :]).astype(int)
        cols = np.round(points[0, :]).astype(int)

        # Clip to image bounds
        rows = np.clip(rows, 0, img.shape[0] - 1)
        cols = np.clip(cols, 0, img.shape[1] - 1)

        intensity = img[rows, cols, :] / 255.0
        
        total_lidar_points.append(lidar_points.T[:, :3])
        total_lidar_colors.append(intensity)
        
    total_lidar_points = np.concatenate(total_lidar_points, axis=0)
    total_lidar_colors = np.concatenate(total_lidar_colors, axis=0)
    
    # mapping image to lidar
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(total_lidar_points)
    pcd.colors = o3d.utility.Vector3dVector(total_lidar_colors)
    
    o3d.visualization.draw_geometries([pcd])
        
        

def pointcloud_to_image(sample_token: str,
                        min_dist: float = 1.0):
    camera_sensors = {
        'CAM_FRONT',
        'CAM_FRONT_LEFT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT'
    }
    sensor = 'LIDAR_TOP'
    
    cur_sample = nusc.get('sample', sample_token)
    
    for camera_sensor in camera_sensors:
        # camera
        cam_rec = nusc.get('sample_data', cur_sample['data'][camera_sensor])
        cs_rec = nusc.get('calibrated_sensor', cam_rec['calibrated_sensor_token'])
        ego2cam = transform_matrix(cs_rec['translation'], Quaternion(cs_rec['rotation']), inverse=False)
            
        dist_intrinsic = np.array(cs_rec['camera_intrinsic'])
        distortion = np.array(cs_rec['distortion'])
        
        # lidar
        lidar_rec = nusc.get('sample_data', cur_sample['data'][sensor])
        cs_rec = nusc.get('calibrated_sensor', lidar_rec['calibrated_sensor_token'])
        lidar2ego = transform_matrix(cs_rec['translation'], Quaternion(cs_rec['rotation']), inverse=True) # inverse false is ego2lidar
        
        # sensor
        img_path = osp.join(dataroot, cam_rec['filename'])
        pcl_path = osp.join(dataroot, lidar_rec['filename'])

        img = np.array(Image.open(img_path))
        lidar = LidarPointCloud.from_file(pcl_path)
        
        h, w = img.shape[:2]

        if (camera_sensor != 'CAM_FRONT'):
            balance = 0.0  # 0: maximum crop, 1: maximum FOV
            intrinsic = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                dist_intrinsic, distortion, (w, h), np.eye(3), balance=balance
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                dist_intrinsic, distortion, np.eye(3), intrinsic, (w, h), cv2.CV_16SC2
            )
            img = cv2.remap(img, map1, map2,
                                    interpolation=cv2.INTER_LINEAR)
        else:
            # print(img_path)
            # print(pcl_path)
            intrinsic, _ = cv2.getOptimalNewCameraMatrix(dist_intrinsic, distortion, (w, h), 0)
            img = cv2.undistort(img, dist_intrinsic, distortion, None, intrinsic)
        
        lidar.transform(lidar2ego)
        lidar.transform(ego2cam)
        
        depths = lidar.points[2, :]
        normalized_points = lidar.points[:3, :] / depths
        points = intrinsic @ normalized_points
        
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < w - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < h - 1)
        
        points = points[:, mask]
        coloring = depths[mask]
        
        u = points[0, :].astype(np.int32)
        v = points[1, :].astype(np.int32)
                
        grayscale = ((coloring - coloring.min()) /
                    (coloring.ptp() + 1e-6) * 255).astype(np.uint8)   # (N,)

        jet_colors = cv2.applyColorMap(grayscale.reshape(-1, 1),      # (N, 1)
                                    cv2.COLORMAP_JET)              # (N, 1, 3) BGR

        # Flatten from (N, 1, 3) to (N, 3).
        jet_colors = jet_colors[:, 0, :]                              # (N, 3)

        u = points[0, :].astype(np.int32)   # x‑pixel coordinates
        v = points[1, :].astype(np.int32)   # y‑pixel coordinates

        img_vis = img.copy()
        for i in range(points.shape[1]):
            color = tuple(int(c) for c in jet_colors[i])  # BGR tuple (uint8)
            cv2.circle(img_vis, (u[i], v[i]), 1, color, -1)  # radius 2 px, filled

        cv2.imshow(camera_sensor, cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)            # Wait for any key press
        cv2.destroyAllWindows()   # Close the OpenCV window

cur_scene = nusc.scene[7]

pointcloud_to_image(cur_scene['first_sample_token'])
image_to_pointcloud(cur_scene['first_sample_token'])