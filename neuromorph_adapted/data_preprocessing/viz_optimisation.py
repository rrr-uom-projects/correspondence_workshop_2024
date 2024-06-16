import open3d as o3d
import numpy as np
import os
import pyvista
from tqdm import tqdm
from edge_functions import split_long_edges, collapse_short_edges

def getFiles(targetdir):
    ls = []
    for fname in os.listdir(targetdir):
        path = os.path.join(targetdir, fname)
        if os.path.isdir(path):
            continue
        ls.append(fname)
    return sorted(ls)

def pyvistarise(points, faces):
    return pyvista.PolyData(points, np.insert(faces, 0, 3, axis=1), deep=True, n_faces=len(faces))


source_dir_optimesh = "M:/meshes_for_viz/rigid_reg_zero_CoM_lung_cut_meshes/"
source_dir_no_optim = "M:/meshes_for_viz/no_optimesh/"
source_dir_custom_optim = "M:/meshes_for_viz/custom_optimisation/"

pat_fnames = sorted(getFiles(source_dir_optimesh))

idx = 200
idx = 111
idx = 5
#idx = 45

pat_fname = pat_fnames[idx]
print(pat_fname)

mesh_optimesh = o3d.io.read_triangle_mesh(os.path.join(source_dir_optimesh, pat_fname))
mesh_no_optim = o3d.io.read_triangle_mesh(os.path.join(source_dir_no_optim, pat_fname))
mesh_custom_optim = o3d.io.read_triangle_mesh(os.path.join(source_dir_custom_optim, pat_fname))

points_optim = np.asarray(mesh_optimesh.vertices)
faces_optim = np.asarray(mesh_optimesh.triangles)
points_no_optim = np.asarray(mesh_no_optim.vertices)[:] + np.array([0,0,60])
faces_no_optim = np.asarray(mesh_no_optim.triangles)
points_custom_optim = np.asarray(mesh_custom_optim.vertices)[:] + np.array([0,0,120])
faces_custom_optim = np.asarray(mesh_custom_optim.triangles)

pyv_mesh_optim = pyvistarise(points_optim, faces_optim)
pyv_mesh_no_optim = pyvistarise(points_no_optim, faces_no_optim)
pyv_mesh_custom_optim = pyvistarise(points_custom_optim, faces_custom_optim)

# show mesh qulaity
qual_optim = pyv_mesh_optim.compute_cell_quality(quality_measure='scaled_jacobian')
qual_no_optim = pyv_mesh_no_optim.compute_cell_quality(quality_measure='scaled_jacobian')
qual_custom_optim = pyv_mesh_custom_optim.compute_cell_quality(quality_measure='scaled_jacobian')

pyvista.global_theme.background = 'white'
plotter = pyvista.Plotter()
#plotter.store_image = True
#plotter.add_mesh(pyv_mesh_optim, show_edges=True, line_width=1, edge_color=[0,0,0,1], color=[0, 0, 0.75])
#plotter.add_mesh(pyv_mesh_no_optim, show_edges=True, line_width=1, edge_color=[0,0,0,1], color=[0, 1., 0])
#plotter.add_mesh(pyv_mesh_custom_optim, show_edges=True, line_width=1, edge_color=[0,0,0,1], color=[1., 0, 0])

sargs = dict(title_font_size=28, label_font_size=26, shadow=False, n_labels=5, italic=False, fmt="%.1f", font_family="arial", color='black', bold=False)
plotter.add_mesh(qual_optim, scalars="CellQuality", scalar_bar_args=sargs)
plotter.add_mesh(qual_no_optim, scalars="CellQuality", scalar_bar_args=sargs)
plotter.add_mesh(qual_custom_optim, scalars="CellQuality", scalar_bar_args=sargs)

# add text labels
y = np.max(points_optim[:,0] + 5)
poly = pyvista.PolyData([np.array([y, 0., 10.]), np.array([y, 0., 70.]), np.array([y, 0., 130.])])
poly["My Labels"] = ["Optimesh", "No optimisation", "Custom optimisation"]
plotter.add_point_labels(poly, "My Labels", point_size=0, font_size=28, always_visible=True, text_color='k', font_family="arial", shape=None, bold=False)
plotter.enable_parallel_projection()
camera_pos = plotter.show(window_size=[1600, 1000], auto_close=True)