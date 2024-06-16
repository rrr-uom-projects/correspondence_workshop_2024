# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from os.path import join
import torch.utils.data
import open3d as o3d
from neuromorph_adapted.utils.shape_utils import *
from tqdm import tqdm

def batch_to_shape(batch):
    shape = Shape(batch["verts"].squeeze().to(device), batch["triangles"].squeeze().to(device, torch.long) )

    if "D" in batch:
        shape.D = batch["D"].squeeze().to(device)

    if "sub" in batch:
        shape.sub = batch["sub"]
        for i_s in range(len(shape.sub)):
            for i_p in range(len(shape.sub[i_s])):
                shape.sub[i_s][i_p] = shape.sub[i_s][i_p].to(device)

    if "idx" in batch:
        shape.samples = batch["idx"].squeeze().to(device, torch.long)

    if "verts_full" in batch:
        shape.vert_full = batch["verts_full"].squeeze().to(device)

    return shape

class bilateral_dataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, fnames, load_dist_mat, axis=1):
        self.folder_path = data_folder
        self.fnames = fnames
        self.load_dist_mat = load_dist_mat
        self.axis = axis
        self.data_l = []
        self.data_r = []

        self._init_data()

        self.num_shapes_l = len(self.data_l)
        self.num_shapes_r = len(self.data_r)
        self.num_pairs_l = self.num_shapes_l**2
        self.num_pairs_r = self.num_shapes_r**2
        self.num_pairs = self.num_pairs_l + self.num_pairs_r
        

    def dataset_name_str(self):
        return "bilateral_dataset"

    def _init_data(self):
        print("Loading data...")
        for fname in tqdm(self.fnames):
            file_name = join(self.folder_path, "meshes/", fname + ".ply")
            if "parotid_lt" in file_name or "submandibular_lt" in file_name:
                l_or_r = "l"
            elif "parotid_rt" in file_name or "submandibular_rt" in file_name:
                l_or_r = "r"
            else:
                raise ValueError("file name not recognized")

            load_data = o3d.io.read_triangle_mesh(file_name)

            data_curr = {}
            data_curr["fname"] = fname
            data_curr["verts"] = np.asarray(load_data.vertices).astype(np.float32)
            data_curr["triangles"] = np.asarray(load_data.triangles)

            if self.load_dist_mat:
                file_name = join(self.folder_path, "geodesic_distances/", fname + ".npy")
                D = np.load(file_name)
                D[D > 1e2] = 2
                data_curr["D"] = D

            if l_or_r == "l":
                self.data_l.append(data_curr)
            else:
                self.data_r.append(data_curr)

    def __getitem__(self, index):
        if index < self.num_pairs_l:
            i1 = int(index / self.num_shapes_l)
            i2 = int(index % self.num_shapes_l)
            data_curr = dict()
            data_curr["X"] = self.data_l[i1]
            data_curr["Y"] = self.data_l[i2]
            data_curr["axis"] = self.axis
            return data_curr
        index -= self.num_pairs_l
        i1 = int(index / self.num_shapes_r)
        i2 = int(index % self.num_shapes_r)
        data_curr = dict()
        data_curr["X"] = self.data_r[i1]
        data_curr["Y"] = self.data_r[i2]
        data_curr["axis"] = self.axis
        return data_curr

    def __len__(self):
        return self.num_pairs

class general_dataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, fnames, load_dist_mat, axis=1):
        self.folder_path = data_folder
        self.fnames = fnames
        self.load_dist_mat = load_dist_mat
        self.axis = axis
        self.data = []

        self._init_data()

        self.num_shapes = len(self.data)
        self.num_pairs = self.num_shapes**2
        

    def dataset_name_str(self):
        return "general_dataset"

    def _init_data(self):
        print("Loading data...")
        for fname in tqdm(self.fnames):
            file_name = join(self.folder_path, "meshes/", fname + ".obj")
            
            load_data = o3d.io.read_triangle_mesh(file_name)

            data_curr = {}
            data_curr["fname"] = fname
            data_curr["verts"] = np.asarray(load_data.vertices).astype(np.float32)
            data_curr["triangles"] = np.asarray(load_data.triangles)

            if self.load_dist_mat:
                file_name = join(self.folder_path, "geodesic_distances/", fname + ".npy")
                D = np.load(file_name).astype(np.float32)
                D[D > 1e2] = 2
                data_curr["D"] = D

            self.data.append(data_curr)

    def __getitem__(self, index):
        i1 = int(index / self.num_shapes)
        i2 = int(index % self.num_shapes)
        data_curr = dict()
        data_curr["X"] = self.data[i1]
        data_curr["Y"] = self.data[i2]
        data_curr["axis"] = self.axis
        return data_curr

    def __len__(self):
        return self.num_pairs