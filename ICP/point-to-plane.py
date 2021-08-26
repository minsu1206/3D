import open3d as o3d
import numpy as np
import os
import glob
import random
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

# Prepare bun000.ply, bun045.ply


def visualize(s, t):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    # for s in src:
    ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=2, color='g')
    # for t in tar:
    ax.scatter(t[:, 0], t[:, 1], t[:, 2], s=2, color='r')
    plt.show()


def normal_vector_set(pc):
    tree = KDTree(pc, leaf_size=2)
    dist, ind = tree.query(pc, k=3)
    n_vectors = []
    for i in range(len(pc)):
        v1 = pc[ind[i, 1]] - pc[i]
        v2 = pc[ind[i, 2]] - pc[i]
        n = np.cross(v1, v2)
        n_vectors.append(n)
    return np.array(n_vectors)


def nearest_neighbor(src, dst):
    tree = KDTree(src, leaf_size=2)
    dist, ind = tree.query(dst, k=1)
    src_neighbored = src[ind].reshape(-1, 3)
    return src_neighbored, dist


def point2plane(src, dst, norms):
    N = src.shape[0]
    matrix_b = np.zeros((N, 1))
    for i in range(N):
        matrix_b[i] += norms[i, 0]*dst[i, 0] + norms[i, 1]*dst[i, 1] + norms[i, 2]*dst[i, 2]
        matrix_b[i] -= norms[i, 0]*src[i, 0] + norms[i, 1]*src[i, 1] + norms[i, 2]*dst[i, 2]
    matrix_A = np.zeros((N, 6))
    matrix_A[:, 3:] = norms
    for i in range(N):
        matrix_A[i, :3] = np.cross(src[i], norms[i])
        # matrix_A[i, 0] = norms[i, 2]*src[i, 1] - norms[i, 1]*src[i, 2]
        # matrix_A[i, 1] = norms[i, 0]*src[i, 2] - norms[i, 2]*src[i, 0]
        # matrix_A[i, 2] = norms[i, 1]*src[i, 0] - norms[i, 0]*src[i, 1]
    # U, S, Vt = np.linalg.svd(matrix_A, full_matrices=False)
    # S_inv = np.zeros((U.shape[1], Vt.shape[0]))
    # for i in range(len(S)):
    #     S_inv[i, i] = S[i]
    # S_inv = np.linalg.inv(S_inv)
    # matrix_A_inv = Vt.T.dot(S_inv).dot(U.T)
    matrix_A_inv = np.linalg.pinv(matrix_A)
    x_opt = matrix_A_inv.dot(matrix_b)  # alpha, beta ,gamma, tx, ty, tz
    M = np.eye(4)
    M[0, 1] = -x_opt[2]
    M[0, 2] = x_opt[1]
    M[0, 3] = x_opt[3]
    M[1, 0] = x_opt[2]
    M[1, 2] = -x_opt[0]
    M[1, 3] = x_opt[4]
    M[2, 0] = -x_opt[1]
    M[2, 1] = x_opt[0]
    M[2, 3] = x_opt[3]
    return M


def icp(src, dst, tolerance=1e-7):
    m = src.shape[1]
    assert m == 3
    N = min(src.shape[0], dst.shape[0])
    sampler = random.sample(range(N), N)
    source = np.ones((m+1, N))
    tar = np.ones((m+1, N))
    source[:m, :] = np.copy(src[sampler, :].T)
    tar[:m, :] = np.copy(dst[sampler, :].T)
    prev_error = 0
    count = 0
    M = np.eye(4)
    while True:
        src, dist = nearest_neighbor(src, dst)
        norms = normal_vector_set(dst)
        T = point2plane(src, dst, norms)    # T = 4x4
        source = T.dot(source)
        M = M.dot(T)
        src = source[:3, :].T
        mean_error = np.mean(dist)
        if np.abs(prev_error - mean_error) < tolerance:
            print('Iterate loop:: ', count)
            break
        prev_error = mean_error
        print(prev_error)
        count += 1
    return src, M



if __name__ == '__main__':
    ply_files = sorted(glob.glob(os.path.join('./data', '*.ply')))

    ply_data = [[] for _ in range(len(ply_files))]

    for i, ply in enumerate(ply_files):
        with open(ply, 'r') as ply_:
            lines = ply_.readlines()
            record = False
            for line in lines:
                if record:
                    split_ = line.split(' ')
                    if len(split_) == 4:
                        points = [float(val) for val in line.split(' ')[:-1]]
                        ply_data[i].append(points)
                if 'end_header' in line:
                    record = True

        ply_data[i] = np.array(ply_data[i])
        print(ply_data[i].shape)

    # ply_data[i] = Nx3 format
    # each ply_data[i] has different N. We need to use same amounts.

    for i in range(len(ply_data) -1):
        src_ = ply_data[i]
        tar_ = ply_data[i+1]
        src_result, M = icp(src_, tar_)
        visualize(src_result[:, :3], tar_)
