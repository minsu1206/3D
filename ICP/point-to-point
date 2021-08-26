import open3d as o3d
import numpy as np
import os
import glob
import random
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

## Reference : https://github.com/ClayFlannigan/icp/blob/167cc4adc442487d2b1b331e18818a6f4d779d49/icp.py


# Prepare bun000.ply, bun045.ply


def visualize(s, t):
    plt.clf()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    # for s in src:
    ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=2, color='g')
    # for t in tar:
    ax.scatter(t[:, 0], t[:, 1], t[:, 2], s=2, color='r')
    plt.show()


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''
    # print(A.shape, B.shape)
    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1, :] *= -1
       R = np.dot(Vt.T, U.T)
    # translation
    # s = sum(b.T.dot(a) for a, b in zip(AA, BB)) / sum(a.T.dot(a) for a in AA)
    s=1
    t = centroid_B.T - s * np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t
    # print('BestFitTransform')
    return T, R, t, s


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.0000001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        -> Nx3 array (N은 point 개수)
        B: Nxm numpy array of destination mD point
        -> Nx3 array (N은 point 개수)
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    m = A.shape[1]
    N = min(A.shape[0], B.shape[0])
    size = min(A.shape[0], B.shape[0])
    sampler = random.sample(range(N), size)
    print(len(sampler))
    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1, size))
    dst = np.ones((m+1, size))
    src[:m, :] = np.copy(A[sampler, :].T)
    dst[:m, :] = np.copy(B[sampler, :].T)

    prev_error = 0
    # result_T = np.identity(m+1)
    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        # T, _, _ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)
        T, _, _, s = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        # update the current source
        src = T.dot(src) * s
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    print(mean_error)
    print('calculate final transformation')
    # calculate final transformation
    T, _, _, s = best_fit_transform(A[sampler, :], src[:m, :].T)
    # print(src.shape)
    return T, src.T


if __name__ == '__main__':
    ply_files = sorted(glob.glob(os.path.join('./behind', '*.ply')))
    
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
        T, src_result = icp(src_, tar_)
        # src_homo = np.concatenate((src_, np.ones((len(src_), 1))), axis=1) # Nx4
        # src_result = np.dot(src_homo, T)[:, :3]
        # break
        visualize(src_result[:, :3], tar_)
