import pygeodesic.geodesic as geodesic
import open3d as o3d
import numpy as np
import os
import pyvista
import matplotlib
import matplotlib.pyplot as plt

def getFiles(targetdir):
    ls = []
    for fname in os.listdir(targetdir):
        path = os.path.join(targetdir, fname)
        if os.path.isdir(path):
            continue
        ls.append(fname)
    return sorted(ls)

def pyvistarise(points, faces):
    return pyvista.PolyData(points, np.insert(faces, 0, 3, axis=1), deep=True, n_faces=len(mesh.triangles))

#source_dir = "M:/meshes_for_viz/"
#source_dir = "M:/meshes_for_viz/no_reg/"
source_dir = "M:/meshes_for_viz/rigid_reg_zero_CoM_lung_cut_meshes/"
structs = ["Brainstem", "Spinal-Cord", "Mandible", "Parotid-Lt", "Parotid-Rt", "Submandibular-Lt", "Submandibular-Rt"]
structs = ["Parotid-Lt", "Parotid-Rt"]
for struct in structs:
    pat_fnames = list(sorted(filter(lambda x: struct.lower().replace("-", "_") in x, getFiles(source_dir))))

    plotter = pyvista.Plotter()
    plotter.set_background('white')
    cmap = plt.get_cmap("jet")
    norm = matplotlib.colors.Normalize(vmin=0, vmax=len(pat_fnames))
    scalarMap = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    for pdx, pat_fname in enumerate(pat_fnames):
        print(pdx, pat_fname)
        if pat_fname == "parotid_lt_0522c0457.ply"  or pat_fname == "parotid_lt_0522c0669.ply":
            pass
        else:
            continue

        mesh = o3d.io.read_triangle_mesh(os.path.join(source_dir, pat_fname))
        ydx = pdx // 5
        zdx = pdx % 5
        ydx = pdx
        zdx = 0
        points = np.asarray(mesh.vertices)[:] + np.array([70*zdx, 0, 50*ydx])
        faces = np.asarray(mesh.triangles)
        pyv_mesh = pyvistarise(points, faces)
        plotter.add_mesh(pyv_mesh, show_edges=True, line_width=1, edge_color=[0,0,0,1], color=scalarMap.to_rgba(pdx), opacity=1.)

    camera_pos = plotter.show(window_size=[1600, 1000], auto_close=True)