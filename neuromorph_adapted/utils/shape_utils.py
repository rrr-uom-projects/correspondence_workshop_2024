# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Marvin Eisenberger.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from scipy import sparse
import numpy as np
from torch_geometric.nn import fps, knn_graph
import matplotlib.pyplot as plt
from neuromorph_adapted.param import *
from neuromorph_adapted.utils.base_tools import *


def plot_curr_shape(verts, triangles_x):
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        triangles=triangles_x,
        cmap="viridis",
        linewidths=0.2,
    )
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(-0.4, 0.4)
    ax.set_zlim(-0.4, 0.4)


class Shape:
    """Class for shapes. (Optional) attributes are:
    verts: Vertices in the format nx3
    triangles: Triangles in the format mx3
    samples: Index list of active vertices
    neigh: List of 2-Tuples encoding the adjacency of vertices
    neigh_hessian: Hessian/Graph Laplacian of the shape based on 'neigh'
    mahal_cov_mat: The covariance matrix of our anisotropic arap energy"""

    def __init__(self, verts=None, triangles=None):
        self.verts = verts
        self.triangles = triangles
        self.samples = list(range(verts.shape[0]))
        self.neigh = None
        self.neigh_hessian = None
        self.mahal_cov_mat = None
        self.normal = None
        self.D = None
        self.sub = None
        self.verts_full = None

        if not self.triangles is None:
            self.triangles = self.triangles.to(dtype=torch.long)

    def subsample_fps(self, goal_verts):
        assert (
            goal_verts <= self.verts.shape[0]
        ), "you cannot subsample to more vertices than n"

        ratio = goal_verts / self.verts.shape[0]
        self.samples = fps(self.verts.detach().to(device_cpu), ratio=ratio).to(device)
        self._neigh_knn()

    def reset_sampling(self):
        self.gt_sampling(self.verts.shape[0])

    def gt_sampling(self, n):
        self.samples = list(range(n))
        self.neigh = None

    def scale(self, factor, shift=True):
        self.verts = self.verts * factor

        if shift:
            self.verts = self.verts + (1 - factor) / 2

    def get_bounding_box(self):
        max_x, _ = self.verts.max(dim=0)
        min_x, _ = self.verts.min(dim=0)

        return min_x, max_x

    def to_box(self, shape_y):

        min_x, max_x = self.get_bounding_box()
        min_y, max_y = shape_y.get_bounding_box()

        extent_x = max_x - min_x
        extent_y = max_y - min_y

        self.translate(-min_x)
        shape_y.translate(-min_y)

        scale_fac = torch.max(torch.cat((extent_x, extent_y), 0))
        scale_fac = 1.0 / scale_fac

        self.scale(scale_fac, shift=False)
        shape_y.scale(scale_fac, shift=False)

        extent_x = scale_fac * extent_x
        extent_y = scale_fac * extent_y

        self.translate(0.5 * (1 - extent_x))
        shape_y.translate(0.5 * (1 - extent_y))

    def translate(self, offset):
        self.verts = self.verts + offset.unsqueeze(0)

    def get_verts(self):
        return self.verts[self.samples, :]

    def get_verts_shape(self):
        return self.get_verts().shape

    def get_triangles(self):
        return self.triangles

    def get_triangles_np(self):
        return self.triangles.detach().cpu().numpy()

    def get_verts_np(self):
        return self.verts[self.samples, :].detach().cpu().numpy()

    def get_verts_full_np(self):
        return self.verts.detach().cpu().numpy()

    def get_neigh(self, num_knn=5):
        if self.neigh is None:
            self.compute_neigh(num_knn=num_knn)

        return self.neigh

    def compute_neigh(self, num_knn=5):
        if len(self.samples) == self.verts.shape[0]:
            self._triangles_neigh()
        else:
            self._neigh_knn(num_knn=num_knn)

    def get_edge_index(self, num_knn=5):
        edge_index_one = self.get_neigh(num_knn).t()
        edge_index = torch.zeros(
            [2, edge_index_one.shape[1] * 2], dtype=torch.long, device=self.verts.device
        )
        edge_index[:, : edge_index_one.shape[1]] = edge_index_one
        edge_index[0, edge_index_one.shape[1] :] = edge_index_one[1, :]
        edge_index[1, edge_index_one.shape[1] :] = edge_index_one[0, :]
        return edge_index

    def _triangles_neigh(self):
        self.neigh = torch.cat(
            (self.triangles[:, [0, 1]], self.triangles[:, [0, 2]], self.triangles[:, [1, 2]]), 0
        )

    def _neigh_knn(self, num_knn=5):
        verts = self.get_verts().detach()
        print("Compute knn....")
        self.neigh = (
            knn_graph(verts.to(device_cpu), num_knn, loop=False)
            .transpose(0, 1)
            .to(device)
        )

    def get_neigh_hessian(self):
        if self.neigh_hessian is None:
            self.compute_neigh_hessian()

        return self.neigh_hessian

    def compute_neigh_hessian(self):

        neigh = self.get_neigh()

        n_verts = self.get_verts().shape[0]

        H = sparse.lil_matrix(1e-3 * sparse.identity(n_verts))

        I = np.array(neigh[:, 0].detach().cpu())
        J = np.array(neigh[:, 1].detach().cpu())
        V = np.ones([neigh.shape[0]])
        U = -V
        H = H + sparse.lil_matrix(
            sparse.coo_matrix((U, (I, J)), shape=(n_verts, n_verts))
        )
        H = H + sparse.lil_matrix(
            sparse.coo_matrix((U, (J, I)), shape=(n_verts, n_verts))
        )
        H = H + sparse.lil_matrix(
            sparse.coo_matrix((V, (I, I)), shape=(n_verts, n_verts))
        )
        H = H + sparse.lil_matrix(
            sparse.coo_matrix((V, (J, J)), shape=(n_verts, n_verts))
        )

        self.neigh_hessian = H

    def rotate(self, R):
        self.verts = torch.mm(self.verts, R.transpose(0, 1))

    def to(self, device):
        self.verts = self.verts.to(device)
        self.triangles = self.triangles.to(device)

    def detach_cpu(self):
        self.verts = self.verts.detach().cpu()
        self.triangles = self.triangles.detach().cpu()
        if self.normal is not None:
            self.normal = self.normal.detach().cpu()
        if self.neigh is not None:
            self.neigh = self.neigh.detach().cpu()
        if self.D is not None:
            self.D = self.D.detach().cpu()
        if self.verts_full is not None:
            self.verts_full = self.verts_full.detach().cpu()
        if self.samples is not None and torch.is_tensor(self.samples):
            self.samples = self.samples.detach().cpu()
        if self.sub is not None:
            for i_s in range(len(self.sub)):
                for i_p in range(len(self.sub[i_s])):
                    self.sub[i_s][i_p] = self.sub[i_s][i_p].detach().cpu()

    def compute_volume(self):
        return self.compute_volume_shifted(self.verts)

    def compute_volume_shifted(self, verts_t):
        verts_t = verts_t - verts_t.mean(dim=0, keepdim=True)
        verts_triangles = verts_t[self.triangles, :].to(device_cpu)

        vol_tetrahedra = (verts_triangles.det() / 6).to(device)

        return vol_tetrahedra.sum()

    def get_normal(self):
        if self.normal is None:
            self._compute_outer_normal()
        return self.normal

    def _compute_outer_normal(self):
        edge_1 = torch.index_select(self.verts, 0, self.triangles[:, 1]) - torch.index_select(self.verts, 0, self.triangles[:, 0])
        edge_2 = torch.index_select(self.verts, 0, self.triangles[:, 2]) - torch.index_select(self.verts, 0, self.triangles[:, 0])

        face_norm = torch.cross(1e4 * edge_1, 1e4 * edge_2)

        normal = my_zeros(self.verts.shape)
        for d in range(3):
            normal = torch.index_add(normal, 0, self.triangles[:, d], face_norm)
        self.normal = normal / (1e-5 + normal.norm(dim=1, keepdim=True))


if __name__ == "__main__":
    print("main of shape_utils.py")
