import numpy as np
import open3d as o3d

def faces_to_edge_index(faces):                                 # 0.0095 seconds to 0.0001 seconds
    # edge_index = np.zeros((2, faces.size), dtype=np.int32)
    # for fdx, face in enumerate(faces):
    #     edge_index[:, fdx*3] = np.array([face[0], face[1]])
    #     edge_index[:, fdx*3+1] = np.array([face[1], face[2]])
    #     edge_index[:, fdx*3+2] = np.array([face[2], face[0]])
    # vectorise the above
    edge_index = np.zeros((2, faces.size), dtype=np.int32)
    edge_index[:, 0::3] = faces[:, [0, 1]].T
    edge_index[:, 1::3] = faces[:, [1, 2]].T
    edge_index[:, 2::3] = faces[:, [2, 0]].T
    return edge_index

def get_unique_single_edges(edge_index):                        # 0.0142 seconds to 0.0051 seconds
    # for i in range(edge_index.shape[1]):
    #     if edge_index[0, i] > edge_index[1, i]:
    #         edge_index[:, i] = np.flip(edge_index[:, i])
    # edge_index = np.unique(edge_index, axis=1)
    # vectorise the above
    edge_index[:, edge_index[0, :] > edge_index[1, :]] = np.flip(edge_index[:, edge_index[0, :] > edge_index[1, :]], axis=0)
    edge_index = np.unique(edge_index, axis=1)
    return edge_index

def get_edge_distances(verts, edge_index):                      # 0.0158 seconds to 0.0002 seconds
    # edge_distances = np.zeros(edge_index.shape[1])
    # for i in range(edge_index.shape[1]):
    #     edge_distances[i] = np.linalg.norm(verts[edge_index[0, i]] - verts[edge_index[1, i]])
    # vectorise the above
    edge_distances = np.linalg.norm(verts[edge_index[0, :]] - verts[edge_index[1, :]], axis=1)
    return edge_distances

def find_longest_edge(verts, faces):
    edge_index = faces_to_edge_index(faces)
    unique_edges = get_unique_single_edges(edge_index)
    edge_distances = get_edge_distances(verts, unique_edges)
    longest_edge_index = np.argmax(edge_distances)
    return unique_edges[:, longest_edge_index], edge_distances[longest_edge_index]

def find_shortest_edge(verts, faces):
    edge_index = faces_to_edge_index(faces)
    unique_edges = get_unique_single_edges(edge_index)
    edge_distances = get_edge_distances(verts, unique_edges)
    shortest_edge_index = np.argmin(edge_distances)
    return unique_edges[:, shortest_edge_index], edge_distances[shortest_edge_index]

def split_long_edge(verts, faces, long_edge):
    # find the midpoint of the long edge
    midpoint = (verts[long_edge[0]] + verts[long_edge[1]]) / 2
    # add the midpoint to the verts
    verts = np.vstack((verts, midpoint))
    # find the faces that contain the long edge
    faces_with_long_edge = np.where(np.any(faces == long_edge[0], axis=1) & np.any(faces == long_edge[1], axis=1))[0]
    # split the faces that contain the long edge
    for face in faces[faces_with_long_edge]:
        # find the other vertex in the face
        other_vertex = np.delete(face, np.where(face == long_edge[0]))
        other_vertex = np.delete(other_vertex, np.where(other_vertex == long_edge[1])).item()
        # split the face into two faces
        new_face_1 = np.array([other_vertex, long_edge[0], verts.shape[0]-1])
        new_face_2 = np.array([other_vertex, long_edge[1], verts.shape[0]-1])
        faces = np.vstack((faces, new_face_1))
        faces = np.vstack((faces, new_face_2))
    # remove the old faces
    faces_with_long_edge = sorted(faces_with_long_edge, reverse=True)
    for face in faces_with_long_edge:
        faces = np.delete(faces, face, axis=0)
    return verts, faces

def split_long_edges(verts, faces, iterations=None, max_edge_len=None):
    if iterations is not None:
        for i in range(iterations):
            longest_edge, _ = find_longest_edge(verts, faces)
            verts, faces = split_long_edge(verts, faces, longest_edge)
    elif max_edge_len is not None:
        longest_edge, length = find_longest_edge(verts, faces)
        while length > max_edge_len:
            verts, faces = split_long_edge(verts, faces, longest_edge)
            longest_edge, length = find_longest_edge(verts, faces)
    else:
        raise ValueError("Must provide either iterations or max_edge_len")
    return verts, faces

def remove_unreferenced_verts(verts, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.remove_unreferenced_vertices()
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    return verts, faces

def collapse_short_edge(verts, faces, short_edge):
    # find the midpoint of the short edge
    midpoint = (verts[short_edge[0]] + verts[short_edge[1]]) / 2
    # add the midpoint to the verts
    verts = np.vstack((verts, midpoint))
    # find the faces that contain the short edge
    faces_with_short_edge = np.where(np.any(faces == short_edge[0], axis=1) & np.any(faces == short_edge[1], axis=1))[0]
    # remove faces
    faces_with_short_edge = sorted(faces_with_short_edge, reverse=True)
    for face in faces_with_short_edge:
        faces = np.delete(faces, face, axis=0)
    # identify faces containing the short edge end vertices
    faces_with_old_verts = np.where(np.any(faces == short_edge[0], axis=1) | np.any(faces == short_edge[1], axis=1))[0]
    # replace the old verts with the new vert
    for face in faces_with_old_verts:
        faces[face] = np.where(faces[face] == short_edge[0], verts.shape[0]-1, faces[face])
        faces[face] = np.where(faces[face] == short_edge[1], verts.shape[0]-1, faces[face])
    # remove old verts
    verts, faces = remove_unreferenced_verts(verts, faces)    
    return verts, faces

def collapse_short_edges(verts, faces, iterations=None, min_edge_len=None):
    if iterations is not None:
        for i in range(iterations):
            shortest_edge, _ = find_shortest_edge(verts, faces)
            verts, faces = collapse_short_edge(verts, faces, shortest_edge)
    elif min_edge_len is not None:
        shortest_edge, length = find_shortest_edge(verts, faces)
        while length < min_edge_len:
            verts, faces = collapse_short_edge(verts, faces, shortest_edge)
            shortest_edge, length = find_shortest_edge(verts, faces)
    else:
        raise ValueError("Must provide either iterations or min_edge_len")
    return verts, faces

