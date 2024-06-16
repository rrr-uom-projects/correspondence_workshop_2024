import pygeodesic.geodesic as geodesic
import open3d as o3d
import numpy as np
import os
import pyvista
def pyvistarise(points, faces):
    return pyvista.PolyData(points, np.insert(faces, 0, 3, axis=1), deep=True, n_faces=len(mesh.triangles))

# 1. Open 3D to smooth mesh
# 2. Save as ply
# 3. pygalmesh to remesh
# 4. Simplify to set n_vertices
# 5. Save as ply

source_dir = "C:/PhD/deepmind_surface_meshes/Brainstem/"
source_dir = "M:/meshes_for_viz/"
pat_fname = "0522c0014.ply"
pat_fname = "temp_remesh.ply"
mesh = o3d.io.read_triangle_mesh(os.path.join(source_dir, pat_fname))
# mesh = mesh.filter_smooth_taubin(number_of_iterations=5)
# mesh.remove_unreferenced_vertices()

points = np.asarray(mesh.vertices)
faces = np.asarray(mesh.triangles)

print("Number of vertices: ", points.shape[0])
geoalg = geodesic.PyGeodesicAlgorithmExact(points, faces)

sourceIndex = 1500
targetIndex = 360
distance, path = geoalg.geodesicDistance(sourceIndex, targetIndex)
print(distance)



pyv_mesh = pyvistarise(points, faces)
pyvista.global_theme.background = 'white'
plotter = pyvista.Plotter()
plotter.store_image = True
plotter.add_mesh(pyv_mesh, show_edges=True, line_width=1, edge_color=[0,0,0,1], color=[0.5, 0.706, 1])
plotter.add_lines(path, color='r', width=5)
camera_pos = plotter.show(screenshot="./viz_edges_and_nodes_for_fig.png", window_size=[1600, 1000], auto_close=True)
print(camera_pos)

import matplotlib.pyplot as plt
D = np.load("M:/meshes_for_viz/D.npy")
plt.imshow(D)
plt.colorbar()
assert (D - D.T < 1e-6).all()
plt.show()