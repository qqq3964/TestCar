from tcar import TestCar
from tcar.utils import LidarPointCloud
import os.path as osp
from nuscenes.utils.data_classes import Box
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

dataroot = '/home/taewan/TCAR_DATA' 
nusc = TestCar(version='v1.0-trainval', dataroot=dataroot, verbose=True)

cur_scene = nusc.scene[2]
cur_sample = nusc.get('sample', cur_scene['last_sample_token'])
# next_sample = nusc.get('sample', cur_sample['next'])

def create_3d_bbox(center, size, yaw=0.0):
    # center: [x, y, z], size: [w, l, h]
    w, l, h = size
    bbox = o3d.geometry.OrientedBoundingBox()
    bbox.center = center
    bbox.extent = [w, l, h]
    bbox.R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_axis_angle([0, 0, yaw])
    bbox.color = (1, 0, 0)
    return bbox

def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                     rotation: Quaternion = Quaternion([1, 0, 0, 0]),
                     inverse: bool = False) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation in quaternions (w ri rj rk).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
    tm = np.eye(4)

    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))

    return tm
  
sensor = 'LIDAR_TOP'
lidar_rec = nusc.get('sample_data', cur_sample['data'][sensor])
cs_rec = nusc.get('calibrated_sensor', lidar_rec['calibrated_sensor_token'])
lidar2ego = transform_matrix(cs_rec['translation'], Quaternion(cs_rec['rotation']), inverse=True)

bboxes = []
for ann in cur_sample['anns']:
    rc_ann = nusc.get('sample_annotation', ann)
    bbox = create_3d_bbox(rc_ann['box_center'], rc_ann['size'], yaw=rc_ann['yaw'])
    bboxes.append(bbox)

pcl_path = osp.join(dataroot, lidar_rec['filename'])

lidar = LidarPointCloud.from_file(pcl_path)
lidar.transform(lidar2ego)
lidar_bin = lidar.points.T

intensities = lidar_bin[:, 3]
intensities_normalized = (intensities - intensities.min()) / (intensities.ptp() + 1e-6)  # Avoid divide by zero

colormap = plt.get_cmap("viridis")
colors = colormap(intensities_normalized)[:, :3]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(lidar_bin[:, :3])
pcd.colors = o3d.utility.Vector3dVector(colors)

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
for bbox in bboxes:
    vis.add_geometry(bbox)

render_option = vis.get_render_option()
render_option.point_size = 3.0

vis.run()
vis.destroy_window()