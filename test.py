import numpy as np
import time
import icp
import open3d as o3d

# Constants
N = 10                                    # number of random points in the dataset
num_tests = 100                             # number of test iterations
dim = 3                                     # number of dimensions of the points
noise_sigma = .01                           # standard deviation error to be added
translation = .1                            # max translation of the test set
rotation = .1                               # max rotation (radians) of the test set


def rotation_matrix(axis, theta):
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.)
    b, c, d = -axis*np.sin(theta/2.)

    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                  [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                  [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])


def test_best_fit():

    # Generate a random dataset
    A = np.random.rand(N, dim)

    total_time = 0

    for i in range(num_tests):

        B = np.copy(A)

        # Translate
        t = np.random.rand(dim)*translation
        B += t

        # Rotate
        R = rotation_matrix(np.random.rand(dim), np.random.rand()*rotation)
        B = np.dot(R, B.T).T

        # Add noise
        B += np.random.randn(N, dim) * noise_sigma

        # Find best fit transform
        start = time.time()
        T, R1, t1 = icp.best_fit_transform(B, A)
        total_time += time.time() - start

        # Make C a homogeneous representation of B
        C = np.ones((N, 4))
        C[:,0:3] = B

        # Transform C
        C = np.dot(T, C.T).T

        assert np.allclose(C[:,0:3], A, atol=6*noise_sigma) # T should transform B (or C) to A
        assert np.allclose(-t1, t, atol=6*noise_sigma)      # t and t1 should be inverses
        assert np.allclose(R1.T, R, atol=6*noise_sigma)     # R and R1 should be inverses

    print('best fit time: {:.3}'.format(total_time/num_tests))

    return


def test_icp():

    # Generate a random dataset
    # TODO: +10 will make ICP wrong - need further debugging
    A = np.random.rand(N, dim)+10

    total_time = 0

    for i in range(num_tests):

        B = np.copy(A)

        # Translate
        t = np.random.rand(dim)*translation

        # Rotate
        R = rotation_matrix(np.random.rand(dim), np.random.rand() * rotation)
        B = np.dot(R, B.T-10).T + t + 10

        # Add noise
        # B += np.random.randn(N, dim) * noise_sigma

        # Shuffle to disrupt correspondence
        np.random.shuffle(B)

        # Run ICP
        start = time.time()
        T, distances, iterations = icp.icp(A, B, tolerance=0.000001)
        total_time += time.time() - start

        # Make C a homogeneous representation of B
        C = np.ones((N, 4))
        C[:,0:3] = np.copy(B)

        # Transform C
        C = np.dot(T, C.T).T

        assert np.mean(distances) < 6*noise_sigma                   # mean error should be small
        assert np.allclose(T[0:3,0:3], R, atol=6*noise_sigma)     # T and R should be inverses
        assert np.allclose(T[0:3,3], t, atol=6*noise_sigma)        # T and t should be inverses

    print('icp time: {:.3}'.format(total_time/num_tests))

    return

def test_multi_icp(debug=False):
    # Generate a random dataset
    A1 = np.random.rand(N, dim)
    A2 = np.random.rand(N, dim)+10
    A = np.concatenate((A1, A2), axis=0)
    # np.savetxt('A.txt', A)
    # A = np.loadtxt('A.txt')

    total_time = 0

    for i in range(num_tests):

        B1 = np.copy(A1)
        B2 = np.copy(A2)

        # Translate
        t1 = np.random.rand(dim)*translation
        t2 = -np.random.rand(dim)*translation

        # Rotate
        R1 = rotation_matrix(np.random.rand(dim), np.random.rand() * rotation)
        R2 = rotation_matrix(np.random.rand(dim), np.random.rand() * rotation)
        B1 = np.dot(R1, B1.T).T + t1
        B2 = np.dot(R2, B2.T-10).T + t2 +10
        print("Translate t1: ", t1)
        print("Translate t2: ", t2)
        print("Rotate R1: ", R1)
        print("Rotate R2: ", R2)

        # Add noise
        # B1 += np.random.randn(N, dim) * noise_sigma
        # B2 += np.random.randn(N, dim) * noise_sigma
        B = np.concatenate((B1, B2), axis=0)

        # Shuffle to disrupt correspondence
        # np.random.shuffle(B)
        # np.savetxt('B.txt', B)
        # B = np.loadtxt('B.txt')

        # Run ICP
        start = time.time()
        T_list, distances, iterations, src_corr = icp.multi_icp_2(A, B, tolerance=0.000001)
        total_time += time.time() - start

        # Make C a homogeneous representation of B
        # C = np.ones((N, 4))
        # C[:,0:3] = np.copy(B)

        # Transform C
        # C = np.dot(T, C.T).T

        print("T_list: ", T_list)
        print("iterations: ", iterations)
        print("distances: ", distances)
        print("src_corr: ", src_corr)

        assert np.mean(distances) < 6*noise_sigma                   # mean error should be small
        # assert np.allclose(T_list[0][0:3,0:3].T, R1, atol=6*noise_sigma)     # T and R should be inverses
        # assert np.allclose(T_list[1][0:3,0:3].T, R2, atol=6*noise_sigma)     # T and R should be inverses
        # assert np.allclose(-T_list[0][0:3,3], t1, atol=6*noise_sigma)        # T and t should be inverses
        # assert np.allclose(-T_list[1][0:3,3], t2, atol=6*noise_sigma)        # T and t should be inverses

    print('icp time: {:.3}'.format(total_time/num_tests))

    return

def test_multi_icp_corr():
    # known correspodence from files
    pc0_file = '/home/yixuan/icp/3d_case/3d_case/10655/0.npy'
    pc1_file = '/home/yixuan/icp/3d_case/3d_case/10655/1.npy'

    pc0_cropped = np.load(pc0_file)
    pc1_cropped = np.load(pc1_file)

    # known correspodence test
    n = 3
    T_list, distances, i, src_corr = icp.multi_icp_known_corr(pc0_cropped, pc1_cropped, n=n)
    print(distances.mean())

    # calculate partially warped observed point
    src = np.ones((4,pc0_cropped.shape[0]))
    src[:3,:] = np.copy(pc0_cropped.T)
    for j in range(n):
        src[:, src_corr==j] = np.dot(T_list[j], src[:, src_corr==j])
    warp_pc0_cropped = src[:3,:].T

    warp_color = np.zeros((pc0_cropped.shape[0], 3))
    color_list = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    for i in range(n):
        warp_color[src_corr==i] = color_list[i]

    # visualize warped
    warp_src_o3d = o3d.geometry.PointCloud()
    warp_src_o3d.points = o3d.utility.Vector3dVector(warp_pc0_cropped)
    warp_src_o3d.colors = o3d.utility.Vector3dVector(warp_color)

    o3d.visualization.draw_geometries([warp_src_o3d])
    # o3d.visualization.draw_geometries([pc1_cropped_o3d])
    # o3d.visualization.draw_geometries([warp_pc0_cropped_o3d])
    return

if __name__ == "__main__":
    test_best_fit()
    # test_icp()
    # test_multi_icp()
    test_multi_icp_corr()