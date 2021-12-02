import numpy as np
from sklearn.neighbors import NearestNeighbors


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

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    if A.shape[0] == 0:
        return np.identity(m+1), np.identity(m), np.zeros(m)

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
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)
    # print("A shape: ", A.shape)
    # print("B shape: ", B.shape)
    # print("centroid_A: ", centroid_A)
    # print("centroid_B: ", centroid_B)
    # print("R: ", R)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


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

    # assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    # assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, distances, i

def multi_icp(A, B, n=2, max_iterations=100, tolerance=0.001):
    '''
    The Iterative Closest Point method for multiple rigid objects: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        n: number of objects
        init_pose: list of (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''
    # get number of dimensions
    N, m = A.shape

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # find the nearest neighbors between the current source and destination points
    distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

    dist_list = np.split(distances, n)
    tgt_ind_list = np.split(indices, n)
    src_ind_list = np.split(np.arange(indices.shape[0]), n)

    T_list = []

    for i in range(n):
        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,src_ind_list[i]].T, dst[:m,tgt_ind_list[i]].T)
        T_list.append(T)
    
    prev_error = 0

    for i in range(max_iterations):
        # update correspondence
        for j in range(n):
            src_j = np.dot(T_list[j], src)
            distances, indices = nearest_neighbor(src_j[:m,:].T, dst[:m,:].T)
            dist_list[j] = distances
            tgt_ind_list[j] = indices
        dist_list_arr = np.array(dist_list) # nxN
        tgt_ind_list_arr = np.array(tgt_ind_list) # nxN, closest target indices; Note: multiple transformed source points can have the same closest target
        src_corr = dist_list_arr.argmin(axis=0) # N, src_corr is the index of chosen transformation

        # check error
        mean_error = np.mean(dist_list_arr.min(axis=0))
        if np.abs(prev_error - mean_error) < tolerance:
            break

        # update transformation
        for j in range(n):
            src_subset = src[:m, src_corr==j]
            dst_subset = dst[:m, tgt_ind_list_arr[j][src_corr==j]]
            T,_,_ = best_fit_transform(src_subset.T, dst_subset.T)
            T_list[j] = T
        
        # update source
        for j in range(n):
            src[:, src_corr==j] = np.dot(T_list[j], src[:, src_corr==j])
    return T_list, distances, i