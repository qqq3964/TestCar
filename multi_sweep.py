# from tcar import TestCar
# from tcar.utils import LidarPointCloud
# import os.path as osp
# from nuscenes.utils.data_classes import Box
# import open3d as o3d
# import numpy as np
# import matplotlib.pyplot as plt

# dataroot = '/home/yong/yongdb/dev/LLM-3D_Lane_Labeling_Machine/data/TCAR_DATA'
# nusc = TestCar(version='v1.0-trainval', dataroot=dataroot, verbose=True)

# cur_scene = nusc.scene[20]
# cur_sample = nusc.get('sample', cur_scene['last_sample_token'])

# sensor = 'LIDAR_TOP'
# lidar_data = nusc.get('sample_data', cur_sample['data'][sensor])
# pcl_path = osp.join(dataroot, lidar_data['filename'])
# lidar = LidarPointCloud.from_file(pcl_path)
# all_pc, all_t = LidarPointCloud.from_file_multisample(nusc, cur_sample, sensor, sensor, nsamples=20)
# lidar_bin = all_pc.points.T

# intensities = lidar_bin[:, 3]
# intensities_normalized = (intensities - intensities.min()) / (intensities.ptp() + 1e-6)  # Avoid divide by zero

# colormap = plt.get_cmap("viridis")
# colors = colormap(intensities_normalized)[:, :3]

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(lidar_bin[:, :3])
# pcd.colors = o3d.utility.Vector3dVector(colors)

# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(pcd)

# render_option = vis.get_render_option()
# render_option.point_size = 3.0

# vis.run()
# vis.destroy_window()

from nuscenes.utils.data_classes import Box

from tcar import TestCar
from tcar.utils import LidarPointCloud
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import vtk

dataroot = '/home/yong/yongdb/dev/LLM-3D_Lane_Labeling_Machine/data/TCAR_DATA'
nusc = TestCar(version='v1.0-trainval', dataroot=dataroot, verbose=True)

cur_scene = nusc.scene[20]
cur_sample = nusc.get('sample', cur_scene['last_sample_token'])

sensor = 'LIDAR_TOP'
lidar_data = nusc.get('sample_data', cur_sample['data'][sensor])
pcl_path = osp.join(dataroot, lidar_data['filename'])
lidar = LidarPointCloud.from_file(pcl_path)
all_pc, all_t = LidarPointCloud.from_file_multisample(nusc, cur_sample, sensor, sensor, nsamples=20)
lidar_bin = all_pc.points.T

intensities = lidar_bin[:, 3]
intensities_normalized = (intensities - intensities.min()) / (intensities.ptp() + 1e-6)  # Avoid divide by zero

colormap = plt.get_cmap("viridis")
colors = (colormap(intensities_normalized)[:, :3] * 255).astype(np.uint8)

# --- VTK 시각화 ---
vtk_points = vtk.vtkPoints()
vtk_colors = vtk.vtkUnsignedCharArray()
vtk_colors.SetNumberOfComponents(3)
vtk_colors.SetName("Colors")

for i in range(lidar_bin.shape[0]):
    vtk_points.InsertNextPoint(float(lidar_bin[i, 0]), float(lidar_bin[i, 1]), float(lidar_bin[i, 2]))
    vtk_colors.InsertNextTuple3(int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))

polydata = vtk.vtkPolyData()
polydata.SetPoints(vtk_points)
polydata.GetPointData().SetScalars(vtk_colors)

vertex_filter = vtk.vtkVertexGlyphFilter()
vertex_filter.SetInputData(polydata)
vertex_filter.Update()

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(vertex_filter.GetOutput())

actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetPointSize(2)

renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(0.1, 0.1, 0.1)

render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(1280, 720)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)
render_window.Render()
interactor.Start()