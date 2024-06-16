import numpy as np
import open3d as o3d
import optimesh
import pygeodesic.geodesic as geodesic
import SimpleITK as sitk
from skimage.measure import marching_cubes
import scipy.ndimage as ndimage

import os
from os.path import join
import shutil
from multiprocessing import Process
import warnings
warnings.filterwarnings("ignore")

from utils import getFiles, getDirs

def connected_components(seg):
    # post-processing using scipy.ndimage.label to eliminate extraneous non-connected voxels
    labels, num_features = ndimage.label(input=seg, structure=np.ones((3,3,3)))
    sizes = ndimage.sum(seg, labels, range(num_features+1))
    seg[(labels!=np.argmax(sizes))] = 0
    return seg

def f(fname, optimise=True):
    global source_dir, output_dir, structure_name, outf_name
    print(f"Started {fname}", flush=True)
    # load mask
    mask = sitk.ReadImage(join(source_dir, fname, "segmentations/", structure_name + ".nrrd"))
    spacing = np.array(mask.GetSpacing())[[2,0,1]]
    mask = sitk.GetArrayFromImage(mask)

    # first apply connected components to remove extraneous non-connected voxels
    mask = connected_components(mask)

    ###
    #      THE ABOVE IS A REALLY INTERESTING DISCUSSION POINT
    ###

    # add cc padding for the spinal cord - ensures mesh is closed at both ends - not sure if this is necessary or not
    if structure_name == "Spinal-Cord":
        mask = np.pad(mask, ((1,1), (0,0), (0,0)), mode="constant", constant_values=0)

    # get the vertices and triangles
    vertices, faces, normals, _ = marching_cubes(volume=mask, level=0.49, spacing=spacing)
    
    if structure_name == "Spinal-Cord":
        # Shift vertices to origin CoM in ap and lr direction
        CoM = np.mean(vertices, axis=0)
        vertices = vertices[:] - np.array([0, CoM[1], CoM[2]])
        vertices[:, 0] -= vertices[:, 0].max()
    else:
        # Shift vertices to origin CoM
        vertices = vertices - np.mean(vertices, axis=0)

    # create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

    # Initial smoothing
    mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=3000)
    mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
    mesh.remove_unreferenced_vertices()

    # optimise mesh
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if optimise:
        vertices, triangles = optimesh.optimize_points_cells(vertices, triangles, "CVT (block-diagonal)", 1.0e-5, 3)
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    # get the vertices and triangles
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    n_vertices = vertices.shape[0]

    # compute geodesic distances between all pairs of vertices
    geo_alg = geodesic.PyGeodesicAlgorithmExact(vertices, triangles)
    D = np.zeros((n_vertices, n_vertices), dtype=np.float32)
    for source_index in range(n_vertices):
        D[source_index], _ = geo_alg.geodesicDistances(np.array([source_index]))
    try:
        assert (D - D.T < 1e-4).all()
    except AssertionError: 
        print(f"Assertion error for {fname} - max value: {np.abs(D - D.T).max()}", flush=True)
        # If nan -> It's likely the mesh is not a single connected component -> the orig contour is disconnected
        return

    # save the geodesic distances
    np.save(join(output_dir, "geodesic_distances/", f"{outf_name}_{fname}.npy"), D)

    # save the mesh
    o3d.io.write_triangle_mesh(join(output_dir, "meshes/", f"{outf_name}_{fname}.ply"), mesh)

    # Romeo Dunn
    print(f"Finished {fname}", flush=True)

def __main__():
    global source_dir, output_dir, structure_name, outf_name
    source_dir = "/home/ed/segmentation_work/deepmind_data/data/nrrds/onc_both/"
    output_dir = "/home/ed/correspondence/data/"
    structure_name = "Spinal-Cord"
    outf_name = "spinal_cord"
    fnames = sorted(getDirs(source_dir))

    processes = [Process(target=f, args=(fname,)) for fname in fnames]

    for process in processes:
        process.start()

    # wait for all processes to complete
    for process in processes:
        process.join()

    # check all were created
    processes = []
    for fname in fnames:
        if not os.path.exists(join(output_dir, "meshes/", f"{outf_name}_{fname}.ply")):
            print(f"Mesh not created for {fname} - trying again without optimisation...", flush=True)
            processes.append(Process(target=f, args=(fname, False)))

    if len(processes):
        for process in processes:
            process.start()

        # wait for all processes to complete
        for process in processes:
            process.join()

    for fname in fnames:
        if not os.path.exists(join(output_dir, "meshes/", f"{outf_name}_{fname}.ply")):
            print(f"Mesh still not created for {fname} - big sad...", flush=True)

    # report that all tasks are completed
    print('Done', flush=True)

__main__()