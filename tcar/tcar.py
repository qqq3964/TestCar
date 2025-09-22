import json
import math
import os
import os.path as osp
import sys
import time
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from nuscenes.utils.color_map import get_colormap
import cv2
from pyquaternion import Quaternion
from PIL import Image
import cv2
from nuscenes.utils.geometry_utils import transform_matrix
import numpy as np
# import open3d as o3d
from tcar.utils import LidarPointCloud

class TestCar(NuScenes):
    def __init__(self,
                 version: str = 'v1.0-mini',
                 dataroot: str = '/data/sets/nuscenes',
                 verbose: bool = True,
                 map_resolution: float = 0.1):
        """
        Loads database and creates reverse indexes and shortcuts.
        :param version: Version to load (e.g. "v1.0", ...).
        :param dataroot: Path to the tables and data.
        :param verbose: Whether to print status messages during load.
        :param map_resolution: Resolution of maps (meters).
        """
        self.version = version
        self.dataroot = dataroot
        self.verbose = verbose
        self.table_names = ['sensor', 'calibrated_sensor', 'ego_pose', 'scene', 'sample', 'sample_data', 'sample_annotation']

        assert osp.exists(self.table_root), 'Database version not found: {}'.format(self.table_root)
        
        start_time = time.time()
        if verbose:
            print("======\nLoading NuScenes tables for version {}...".format(self.version))

        # Explicitly assign tables to help the IDE determine valid class members.
        self.sensor = self.__load_table__('sensor')
        self.calibrated_sensor = self.__load_table__('calibrated_sensor')
        self.ego_pose = self.__load_table__('ego_pose')
        self.scene = self.__load_table__('scene')
        self.sample = self.__load_table__('sample')
        self.sample_data = self.__load_table__('sample_data')
        self.sample_annotation = self.__load_table__('sample_annotation')

        # Initialize the colormap which maps from class names to RGB values.
        self.colormap = get_colormap()

        lidar_tasks = [t for t in ['lidarseg', 'panoptic'] if osp.exists(osp.join(self.table_root, t + '.json'))]
        if len(lidar_tasks) > 0:
            self.lidarseg_idx2name_mapping = dict()
            self.lidarseg_name2idx_mapping = dict()
            self.load_lidarseg_cat_name_mapping()
        for i, lidar_task in enumerate(lidar_tasks):
            if self.verbose:
                print(f'Loading nuScenes-{lidar_task}...')
            if lidar_task == 'lidarseg':
                self.lidarseg = self.__load_table__(lidar_task)
            else:
                self.panoptic = self.__load_table__(lidar_task)

            setattr(self, lidar_task, self.__load_table__(lidar_task))
            label_files = os.listdir(os.path.join(self.dataroot, lidar_task, self.version))
            num_label_files = len([name for name in label_files if (name.endswith('.bin') or name.endswith('.npz'))])
            num_lidarseg_recs = len(getattr(self, lidar_task))
            assert num_lidarseg_recs == num_label_files, \
                f'Error: there are {num_label_files} label files but {num_lidarseg_recs} {lidar_task} records.'
            self.table_names.append(lidar_task)
            # Sort the colormap to ensure that it is ordered according to the indices in self.category.
            self.colormap = dict({c['name']: self.colormap[c['name']]
                                  for c in sorted(self.category, key=lambda k: k['index'])})

        # If available, also load the image_annotations table created by export_2d_annotations_as_json().
        if osp.exists(osp.join(self.table_root, 'image_annotations.json')):
            self.image_annotations = self.__load_table__('image_annotations')

        if verbose:
            for table in self.table_names:
                print("{} {},".format(len(getattr(self, table)), table))
            print("Done loading in {:.3f} seconds.\n======".format(time.time() - start_time))
            
        # Make reverse indexes for common lookups.
        self.__make_reverse_index__(verbose)
        
        # Initialize NuScenesExplorer class.
        self.explorer = NuScenesExplorer(self)

    def __make_reverse_index__(self, verbose: bool) -> None:
        """
        De-normalizes database to create reverse indices for common cases.
        :param verbose: Whether to print outputs.
        """

        start_time = time.time()
        if verbose:
            print("Reverse indexing ...")

        # Store the mapping from token to table index for each table.
        self._token2ind = dict()
        for table in self.table_names:
            self._token2ind[table] = dict()

            for ind, member in enumerate(getattr(self, table)):
                self._token2ind[table][member['token']] = ind

        # Decorate (adds short-cut) sample_data with sensor information.
        for record in self.sample_data:
            cs_record = self.get('calibrated_sensor', record['calibrated_sensor_token'])
            sensor_record = self.get('sensor', cs_record['sensor_token'])
            record['sensor_modality'] = sensor_record['modality']
            record['channel'] = sensor_record['channel']

        # Reverse-index samples with sample_data and annotations.
        for record in self.sample:
            record['data'] = {}
            record['anns'] = []

        for record in self.sample_data:
            if record['is_key_frame']:
                sample_record = self.get('sample', record['sample_token'])
                sample_record['data'][record['channel']] = record['token']

        for idx, ann_record in enumerate(self.sample_annotation):
            sample_record = self.get('sample', ann_record['sample_token'])
            sample_record['anns'].append(ann_record['token'])

        if verbose:
            print("Done reverse indexing in {:.1f} seconds.\n======".format(time.time() - start_time))
        
    def image_to_pointcloud(self,
                            sample_token: str,
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
        
        cur_sample = self.get('sample', sample_token)
        
        total_lidar_points = list()
        total_lidar_colors = list()
        for camera_sensor in camera_sensors:
            # camera
            cam_rec = self.get('sample_data', cur_sample['data'][camera_sensor])
            cs_rec = self.get('calibrated_sensor', cam_rec['calibrated_sensor_token'])
            ego2cam = transform_matrix(cs_rec['translation'], Quaternion(cs_rec['rotation']), inverse=False)
                
            dist_intrinsic = np.array(cs_rec['camera_intrinsic'])
            distortion = np.array(cs_rec['distortion'])
            
            # lidar
            lidar_rec = self.get('sample_data', cur_sample['data'][sensor])
            cs_rec = self.get('calibrated_sensor', lidar_rec['calibrated_sensor_token'])
            lidar2ego = transform_matrix(cs_rec['translation'], Quaternion(cs_rec['rotation']), inverse=True) # inverse false is ego2lidar
            
            # sensor
            img_path = osp.join(self.dataroot, cam_rec['filename'])
            pcl_path = osp.join(self.dataroot, lidar_rec['filename'])

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
        
        # # mapping image to lidar
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(total_lidar_points)
        # pcd.colors = o3d.utility.Vector3dVector(total_lidar_colors)
        
        # o3d.visualization.draw_geometries([pcd])
            

    def pointcloud_to_image(self,
                            sample_token: str,
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
        
        cur_sample = self.get('sample', sample_token)
        
        for camera_sensor in camera_sensors:
            # camera
            cam_rec = self.get('sample_data', cur_sample['data'][camera_sensor])
            cs_rec = self.get('calibrated_sensor', cam_rec['calibrated_sensor_token'])
            ego2cam = transform_matrix(cs_rec['translation'], Quaternion(cs_rec['rotation']), inverse=False)
                
            dist_intrinsic = np.array(cs_rec['camera_intrinsic'])
            distortion = np.array(cs_rec['distortion'])
            
            # lidar
            lidar_rec = self.get('sample_data', cur_sample['data'][sensor])
            cs_rec = self.get('calibrated_sensor', lidar_rec['calibrated_sensor_token'])
            lidar2ego = transform_matrix(cs_rec['translation'], Quaternion(cs_rec['rotation']), inverse=True) # inverse false is ego2lidar
            
            # sensor
            img_path = osp.join(self.dataroot, cam_rec['filename'])
            pcl_path = osp.join(self.dataroot, lidar_rec['filename'])

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