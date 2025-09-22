import copy
import os.path as osp
import struct
from abc import ABC, abstractmethod
from functools import reduce
from typing import Tuple, List, Dict

import cv2
import numpy as np
from matplotlib.axes import Axes
from pyquaternion import Quaternion

from nuscenes.lidarseg.lidarseg_utils import colormap_to_colors, create_lidarseg_legend
from nuscenes.utils.data_io import load_bin_file
from nuscenes.utils.geometry_utils import view_points, transform_matrix
# import open3d as o3d



class PointCloud(ABC):
    """
    Abstract class for manipulating and viewing point clouds.
    Every point cloud (lidar and radar) consists of points where:
    - Dimensions 0, 1, 2 represent x, y, z coordinates.
        These are modified when the point cloud is rotated or translated.
    - All other dimensions are optional. Hence these have to be manually modified if the reference frame changes.
    """

    def __init__(self, points: np.ndarray):
        """
        Initialize a point cloud and check it has the correct dimensions.
        :param points: <np.float: d, n>. d-dimensional input point cloud matrix.
        """
        assert points.shape[0] == self.nbr_dims(), 'Error: Pointcloud points must have format: %d x n' % self.nbr_dims()
        self.points = points

    @staticmethod
    @abstractmethod
    def nbr_dims() -> int:
        """
        Returns the number of dimensions.
        :return: Number of dimensions.
        """
        pass

    @classmethod
    @abstractmethod
    def from_file(cls, file_name: str) -> 'PointCloud':
        """
        Loads point cloud from disk.
        :param file_name: Path of the pointcloud file on disk.
        :return: PointCloud instance.
        """
        pass
    
    # def convert_to_o3d(self):
    #     pc = o3d.geometry.PointCloud()
    #     xyz = self.points[:3, :].T
    #     pc.points = o3d.utility.Vector3dVector(xyz)

    #     # intensity
    #     if self.points.shape[0] >= 4:
    #         intensity = self.points[3, :]  # shape: (N,)
    #         # Normalize intensity to 0~1 range (for visualization)
    #         intensity = intensity / 255.0
    #         colors = np.tile(intensity[:, np.newaxis], (1, 3))  # shape: (N, 3)
    #         pc.colors = o3d.utility.Vector3dVector(colors)

    #     return pc
    
    @staticmethod
    def convert_from_o3d(pc):
        points = np.asarray(pc.points).T  # shape: (3, N)
        if pc.has_colors():
            colors = np.asarray(pc.colors) * 255.0 # shape: (N, 3)
            intensity = colors.mean(axis=1, keepdims=True).T  # shape: (1, N)
        else:
            # No color → dummy intensity
            intensity = np.zeros((1, points.shape[1]), dtype=points.dtype)

        # Stack as 4xN
        points_with_intensity = np.vstack((points, intensity))  # shape: (4, N)
        
        return points_with_intensity
    
    # @classmethod
    # def from_file_multisample(cls,
    #                         nusc: 'NuScenes',
    #                         sample_rec: Dict,
    #                         chan: str,
    #                         ref_chan: str,
    #                         nsamples: int = 5,
    #                         min_distance: float = 1.0) -> Tuple['PointCloud', np.ndarray]:

    #     points = np.zeros((cls.nbr_dims(), 0), dtype=np.float32 if cls == LidarPointCloud else np.float64)
    #     all_pc = cls(points)
    #     all_times = np.zeros((1, 0))

    #     ref_sd_token = sample_rec['data'][ref_chan]
    #     ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    #     ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    #     ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    #     ref_time = 1e-6 * ref_sd_rec['timestamp']

    #     sample_data_token = sample_rec['data'][chan]
    #     current_sd_rec = nusc.get('sample_data', sample_data_token)

    #     prev_pc = None

    #     for i in range(nsamples):
    #         current_pc = cls.from_file(osp.join(nusc.dataroot, current_sd_rec['filename']))
    #         current_pc.remove_close(min_distance)

    #         current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
    #         global_from_car = transform_matrix(current_pose_rec['translation'], Quaternion(current_pose_rec['rotation']), inverse=False)
    #         current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
    #         car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']), inverse=True)

    #         # pose transformation
    #         ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)
    #         car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=True)
    #         trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
    #         current_pc.transform(trans_matrix)

    #         if prev_pc is not None:
    #             src = current_pc.convert_to_o3d()
    #             tgt = prev_pc.convert_to_o3d()

    #             src = src.voxel_down_sample(voxel_size=0.2)
    #             tgt = tgt.voxel_down_sample(voxel_size=0.2)

    #             reg = o3d.pipelines.registration.registration_generalized_icp(
    #                 src, tgt, max_correspondence_distance=1.0,
    #                 init=np.eye(4),
    #                 estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
    #                 criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))

    #             aligned_src = src.transform(reg.transformation)
    #             aligned_np = cls.convert_from_o3d(aligned_src)
    #             current_pc.points = aligned_np 

    #         # accumulation
    #         time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
    #         times = time_lag * np.ones((1, current_pc.nbr_points()))
    #         all_times = np.hstack((all_times, times))
    #         all_pc.points = np.hstack((all_pc.points, current_pc.points))

    #         prev_pc = current_pc

    #         if sample_rec['prev'] == '':
    #             break
    #         else:
    #             sample_rec = nusc.get('sample', sample_rec['prev'])
    #             current_sd_rec = nusc.get('sample_data', sample_rec['data'][chan])

    #     return all_pc, all_times

    # @classmethod
    # def from_file_multisample(cls,
    #                          nusc: 'NuScenes',
    #                          sample_rec: Dict,
    #                          chan: str,
    #                          ref_chan: str,
    #                          nsamples: int = 5,
    #                          min_distance: float = 1.0) -> Tuple['PointCloud', np.ndarray]:
    #     # Init.
    #     points = np.zeros((cls.nbr_dims(), 0), dtype=np.float32 if cls == LidarPointCloud else np.float64)
    #     all_pc = cls(points)
    #     all_times = np.zeros((1, 0))

    #     # Get reference pose and timestamp.
    #     ref_sd_token = sample_rec['data'][ref_chan]
    #     ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    #     ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    #     ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    #     ref_time = 1e-6 * ref_sd_rec['timestamp']

    #     # Homogeneous transform from ego car frame to reference frame.
    #     ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)

    #     # Homogeneous transformation matrix from global to _current_ ego car frame.
    #     car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=True)

    #     # Aggregate current and previous sweeps.
    #     sample_data_token = sample_rec['data'][chan]
    #     current_sd_rec = nusc.get('sample_data', sample_data_token)
        
    #     for _ in range(nsamples):
    #         # Load up the pointcloud and remove points close to the sensor.
    #         current_pc = cls.from_file(osp.join(nusc.dataroot, current_sd_rec['filename']))
    #         current_pc.remove_close(min_distance)
            
    #         # Get past pose, ego2global
    #         current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
            
    #         global_from_car = transform_matrix(current_pose_rec['translation'], Quaternion(current_pose_rec['rotation']), inverse=False)

    #         # Homogeneous transformation matrix from sensor coordinate frame to ego car frame., ego to lidar
    #         current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
    #         car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
    #                                             inverse=True)
    #         # Fuse four transformation matrices into one and perform transform.
    #         trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
    #         current_pc.transform(trans_matrix)

    #         # Add time vector which can be used as a temporal feature.
    #         time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # Positive difference.
    #         times = time_lag * np.ones((1, current_pc.nbr_points()))
    #         all_times = np.hstack((all_times, times))

    #         # Merge with key pc.
    #         all_pc.points = np.hstack((all_pc.points, current_pc.points))

    #         # Abort if there are no previous sweeps.
    #         if sample_rec['prev'] == '':
    #             break
    #         else:
    #             sample_rec = nusc.get('sample', sample_rec['prev'])
    #             current_sd_rec = nusc.get('sample_data', sample_rec['data'][chan])

    #     return all_pc, all_times
          
        
    @classmethod
    def from_file_multisweep(cls,
                             nusc: 'NuScenes',
                             sample_rec: Dict,
                             chan: str,
                             ref_chan: str,
                             nsweeps: int = 5,
                             min_distance: float = 1.0) -> Tuple['PointCloud', np.ndarray]:
        """
        Return a point cloud that aggregates multiple sweeps.
        As every sweep is in a different coordinate frame, we need to map the coordinates to a single reference frame.
        As every sweep has a different timestamp, we need to account for that in the transformations and timestamps.
        :param nusc: A NuScenes instance.
        :param sample_rec: The current sample.
        :param chan: The lidar/radar channel from which we track back n sweeps to aggregate the point cloud.
        :param ref_chan: The reference channel of the current sample_rec that the point clouds are mapped to.
        :param nsweeps: Number of sweeps to aggregated.
        :param min_distance: Distance below which points are discarded.
        :return: (all_pc, all_times). The aggregated point cloud and timestamps.
        """
        # Init.
        points = np.zeros((cls.nbr_dims(), 0), dtype=np.float32 if cls == LidarPointCloud else np.float64)
        all_pc = cls(points)
        all_times = np.zeros((1, 0))

        # Get reference pose and timestamp.
        ref_sd_token = sample_rec['data'][ref_chan]
        ref_sd_rec = nusc.get('sample_data', ref_sd_token)
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        # Homogeneous transform from ego car frame to reference frame.
        ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)

        # Homogeneous transformation matrix from global to _current_ ego car frame.
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=True)

        # Aggregate current and previous sweeps.
        sample_data_token = sample_rec['data'][chan]
        current_sd_rec = nusc.get('sample_data', sample_data_token)
        for _ in range(nsweeps):
            # Load up the pointcloud and remove points close to the sensor.
            current_pc = cls.from_file(osp.join(nusc.dataroot, current_sd_rec['filename']))
            current_pc.remove_close(min_distance)

            # Get past pose.
            current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
            global_from_car = transform_matrix(current_pose_rec['translation'], Quaternion(current_pose_rec['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
            current_pc.transform(trans_matrix)

            # Add time vector which can be used as a temporal feature.
            time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # Positive difference.
            times = time_lag * np.ones((1, current_pc.nbr_points()))
            all_times = np.hstack((all_times, times))

            # Merge with key pc.
            all_pc.points = np.hstack((all_pc.points, current_pc.points))

            # Abort if there are no previous sweeps.
            if current_sd_rec['prev'] == '':
                break
            else:
                current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

        return all_pc, all_times

    def nbr_points(self) -> int:
        """
        Returns the number of points.
        :return: Number of points.
        """
        return self.points.shape[1]

    def subsample(self, ratio: float) -> None:
        """
        Sub-samples the pointcloud.
        :param ratio: Fraction to keep.
        """
        selected_ind = np.random.choice(np.arange(0, self.nbr_points()), size=int(self.nbr_points() * ratio))
        self.points = self.points[:, selected_ind]

    def remove_close(self, radius: float) -> None:
        """
        Removes point too close within a certain radius from origin.
        :param radius: Radius below which points are removed.
        """

        x_filt = np.abs(self.points[0, :]) < radius
        y_filt = np.abs(self.points[1, :]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        self.points = self.points[:, not_close]

    def translate(self, x: np.ndarray) -> None:
        """
        Applies a translation to the point cloud.
        :param x: <np.float: 3, 1>. Translation in x, y, z.
        """
        for i in range(3):
            self.points[i, :] = self.points[i, :] + x[i]

    def rotate(self, rot_matrix: np.ndarray) -> None:
        """
        Applies a rotation.
        :param rot_matrix: <np.float: 3, 3>. Rotation matrix.
        """
        self.points[:3, :] = np.dot(rot_matrix, self.points[:3, :])

    def transform(self, transf_matrix: np.ndarray) -> None:
        """
        Applies a homogeneous transform.
        :param transf_matrix: <np.float: 4, 4>. Homogenous transformation matrix.
        """
        self.points[:3, :] = transf_matrix.dot(np.vstack((self.points[:3, :], np.ones(self.nbr_points()))))[:3, :]

    def render_height(self,
                      ax: Axes,
                      view: np.ndarray = np.eye(4),
                      x_lim: Tuple[float, float] = (-20, 20),
                      y_lim: Tuple[float, float] = (-20, 20),
                      marker_size: float = 1) -> None:
        """
        Very simple method that applies a transformation and then scatter plots the points colored by height (z-value).
        :param ax: Axes on which to render the points.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min, max). x range for plotting.
        :param y_lim: (min, max). y range for plotting.
        :param marker_size: Marker size.
        """
        self._render_helper(2, ax, view, x_lim, y_lim, marker_size)

    def render_intensity(self,
                         ax: Axes,
                         view: np.ndarray = np.eye(4),
                         x_lim: Tuple[float, float] = (-20, 20),
                         y_lim: Tuple[float, float] = (-20, 20),
                         marker_size: float = 1) -> None:
        """
        Very simple method that applies a transformation and then scatter plots the points colored by intensity.
        :param ax: Axes on which to render the points.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min, max).
        :param y_lim: (min, max).
        :param marker_size: Marker size.
        """
        self._render_helper(3, ax, view, x_lim, y_lim, marker_size)

    def _render_helper(self,
                       color_channel: int,
                       ax: Axes,
                       view: np.ndarray,
                       x_lim: Tuple[float, float],
                       y_lim: Tuple[float, float],
                       marker_size: float) -> None:
        """
        Helper function for rendering.
        :param color_channel: Point channel to use as color.
        :param ax: Axes on which to render the points.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min, max).
        :param y_lim: (min, max).
        :param marker_size: Marker size.
        """
        points = view_points(self.points[:3, :], view, normalize=False)
        ax.scatter(points[0, :], points[1, :], c=self.points[color_channel, :], s=marker_size)
        ax.set_xlim(x_lim[0], x_lim[1])
        ax.set_ylim(y_lim[0], y_lim[1])

class LidarPointCloud(PointCloud):

    @staticmethod
    def nbr_dims() -> int:
        """
        Returns the number of dimensions.
        :return: Number of dimensions.
        """
        return 4

    @classmethod
    def from_file(cls, file_name: str) -> 'LidarPointCloud':
        """
        Loads LIDAR data from binary numpy format. Data is stored as (x, y, z, intensity, ring index).
        :param file_name: Path of the pointcloud file on disk.
        :return: LidarPointCloud instance (x, y, z, intensity).
        """

        assert file_name.endswith('.bin'), 'Unsupported filetype {}'.format(file_name)

        scan = np.fromfile(file_name, dtype=np.float32)
        points = scan.reshape((-1, 4))[:, :cls.nbr_dims()]
        return cls(points.T)
