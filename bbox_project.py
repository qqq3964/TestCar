from tcar import TestCar
from tcar.utils import LidarPointCloud
import os.path as osp
from nuscenes.utils.data_classes import Box
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

TYPE_MAP = {
    0: "UNKNOWN",
    1: "CAR",
    2: "BUS",
    3: "TRUCK",
    4: "CONSTRN_VEH",
    5: "CYC",
    6: "TRICYCLE",
    7: "PED",
    8: "TRAFFIC_CONE",
    9: "BARROW",
    10: "ANIMAL",
    11: "WARN_TRIANGLE",
    12: "BIRD",
    13: "WATER_BARRIER",
    14: "LAMP_POST",
    15: "TRAFFIC_SIGN",
    16: "WARN_POST",
    17: "TRAFFIC_BARREL",
    18: "ARTICULATED_HEAD",
    19: "ARTICULATED_BODY",
    20: "VISION_OBSTACLE",
    50: "STATIC_UNKNOWN"
}

# Mapping from class name to RGB color for bounding box visualization
CLASS_COLOR = {
    "UNKNOWN": (0.5, 0.5, 0.5),           # Gray
    "CAR": (1.0, 0.0, 0.0),               # Red
    "BUS": (0.0, 0.5, 1.0),               # Sky blue
    "TRUCK": (1.0, 0.5, 0.0),             # Orange
    "CONSTRN_VEH": (0.6, 0.4, 0.0),       # Brown
    "CYC": (1.0, 1.0, 0.0),               # Yellow
    "TRICYCLE": (0.7, 1.0, 0.7),          # Light green
    "PED": (1.0, 0.0, 1.0),               # Magenta
    "TRAFFIC_CONE": (1.0, 1.0, 1.0),      # White
    "BARROW": (0.6, 0.3, 0.0),            # Dark orange
    "ANIMAL": (0.8, 0.4, 0.4),            # Light brown
    "WARN_TRIANGLE": (1.0, 0.6, 0.6),     # Light red
    "BIRD": (0.4, 0.4, 1.0),              # Blueish
    "WATER_BARRIER": (0.4, 0.6, 1.0),     # Light blue
    "LAMP_POST": (0.6, 0.6, 0.6),         # Light gray
    "TRAFFIC_SIGN": (1.0, 1.0, 0.6),      # Pale yellow
    "WARN_POST": (1.0, 0.8, 0.6),         # Orange-pink
    "TRAFFIC_BARREL": (1.0, 0.6, 0.2),    # Orange barrel
    "ARTICULATED_HEAD": (0.2, 1.0, 1.0),  # Cyan
    "ARTICULATED_BODY": (0.0, 0.8, 0.8),  # Dark cyan
    "VISION_OBSTACLE": (0.8, 0.0, 0.4),   # Deep magenta
    "STATIC_UNKNOWN": (0.2, 0.2, 0.2)     # Dark gray
}

dataroot = '/home/taewan/TCAR_DATA' 
nusc = TestCar(version='v1.0-trainval', dataroot=dataroot, verbose=True)

cur_scene = nusc.scene[3]
cur_sample = nusc.get('sample', cur_scene['last_sample_token'])
# next_sample = nusc.get('sample', cur_sample['next'])

def parse_type(type_byte: str) -> str:
    """
    Convert single-byte string like '\x03' to class name
    """
    if isinstance(type_byte, str):
        type_int = ord(type_byte)
    elif isinstance(type_byte, bytes):
        type_int = type_byte[0]
    else:
        raise TypeError("Expected str or bytes input.")

    return TYPE_MAP.get(type_int, f"UNKNOWN_TYPE_{type_int}")

def create_3d_bbox(center, size, yaw=0.0, color=(1,0,0)):
    # center: [x, y, z], size: [w, l, h]
    w, l, h = size
    bbox = o3d.geometry.OrientedBoundingBox()
    bbox.center = center
    bbox.extent = [w, l, h]
    bbox.R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_axis_angle([0, 0, yaw])
    bbox.color = color
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
    class_type = parse_type(rc_ann['type'])
    color = CLASS_COLOR.get(class_type, (0.0, 1.0, 0.0))
    bbox = create_3d_bbox(rc_ann['box_center'], rc_ann['size'], yaw=rc_ann['yaw'], color=color)
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