import torch

triv_to_edge = torch.as_tensor([[-1, 1, 0], [0, -1, 1], [1, 0, -1]], dtype=torch.float32)
edge_norm_to_proj = torch.as_tensor([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=torch.float32)

def hat_op(v):
    assert v.shape[1] == 3, "wrong input dimensions"
    w = torch.zeros([3, 3, 3], dtype=torch.float32)
    w[0, 1, 2] = -1
    w[0, 2, 1] = 1
    w[1, 0, 2] = 1
    w[1, 2, 0] = -1
    w[2, 0, 1] = -1
    w[2, 1, 0] = 1
    v = v.transpose(0, 1).unsqueeze(2).unsqueeze(3)
    w = w.unsqueeze(1)
    M = v * w
    M = M.sum(0)
    return M

def cross_prod(u, v):
    if len(v.shape) == 2:
        v = v.unsqueeze(2)
    return torch.bmm(hat_op(u), v)

def soft_relu(m, eps=1e-7):
    return torch.relu(m) + eps

def batch_trace(m):
    m = (m * torch.eye(m.shape[1]).unsqueeze(0)).sum(dim=(1, 2))
    return m.unsqueeze(1).unsqueeze(2)

def discrete_shell_energy_pre(vert_t, vert_0, triv):

    vert_triv_0 = vert_0[triv]  # [n, 3, 3]: #tri x #corner x #dim
    vert_triv_t = vert_t[triv]  # [n, 3, 3]: #tri x #corner x #dim

    edge_0 = 1e2 * torch.matmul(triv_to_edge.unsqueeze(0), vert_triv_0)  # [n, 3, 3]: #tri x #edge x #dim
    edge_t = 1e2 * torch.matmul(triv_to_edge.unsqueeze(0), vert_triv_t)  # [n, 3, 3]: #tri x #edge x #dim

    normal_0 = cross_prod(edge_0[:, 0, :], edge_0[:, 1, :])  # [n, 3, 1]: #tri x #dim x 1
    normal_t = cross_prod(edge_t[:, 0, :], edge_t[:, 1, :])  # [n, 3, 1]: #tri x #dim x 1
    area_0 = soft_relu(normal_0.norm(dim=1, keepdim=True))  # [n, 1, 1]: #tri x 1 x 1
    area_t = soft_relu(normal_t.norm(dim=1, keepdim=True))  # [n, 1, 1]: #tri x 1 x 1

    normal_0 = normal_0 / area_0  # [n, 3, 1]: #tri x #dim x 1
    normal_t = normal_t / area_t  # [n, 3, 1]: #tri x #dim x 1

    edge_proj_0 = cross_prod(normal_0.squeeze(), edge_0.transpose(1, 2)).transpose(1,
                                                                                   2)  # [n, 3, 3]: #tri x #edge x #dim
    edge_proj_0 = edge_proj_0.unsqueeze(2) * edge_proj_0.unsqueeze(3)  # [n, 3, 3, 3]: #tri x #edge x (#dim1 x #dim2)

    return normal_0, normal_t, area_0, area_t, edge_0, edge_t, edge_proj_0

def membrane_transformation(edge_t, area_0, normal_0, edge_proj_0):
    edge_norm_t = torch.norm(edge_t, dim=2, keepdim=True)  # [n, 3, 1]: #tri x #edge x 1
    edge_norm_proj_t = torch.matmul(edge_norm_to_proj.unsqueeze(0), edge_norm_t ** 2).unsqueeze(
        3)  # [n, 3, 1, 1]: #tri x #edge x 1 x 1

    a_membrane = 1 / soft_relu(2 * area_0 ** 2) * torch.sum(edge_norm_proj_t * edge_proj_0,
                                                           dim=1)  # [n, 3, 3]: #tri x (#dim1 x #dim2)

    a_membrane_n = a_membrane + (
            normal_0 * normal_0.transpose(1, 2))  # [n, 3, 3]: #tri x (#dim1 x #dim2)

    return a_membrane, a_membrane_n

def compute_distortion(verts_x, new_verts_x, triangles_x, num_eval=10000):
    dist_max = 100

    num_triangles = triangles_x.shape[0]

    normal_0, _, area_0, _, _, edge_t, edge_proj_0 = discrete_shell_energy_pre(new_verts_x, verts_x, triangles_x)
    _, a_membrane_n = membrane_transformation(edge_t, area_0, normal_0, edge_proj_0)
    distortion_curr = (batch_trace(torch.bmm(a_membrane_n.transpose(1, 2), a_membrane_n)).squeeze()) / \
                        (torch.det(a_membrane_n) + 1e-6) - 3
    distortion_curr = torch.abs(distortion_curr)
    distortion_curr = dist_max - torch.relu(dist_max - distortion_curr)

    idx_eval = torch.zeros([num_eval], dtype=torch.long).random_(0, num_triangles)
    distortion = distortion_curr[idx_eval]

    return distortion