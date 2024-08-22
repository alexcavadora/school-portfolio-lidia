import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import os

def rotate_view(vis):
    ctr = vis.get_view_control()
    ctr.rotate(0.0, 2.0)
    return True

def read_and_clean(file, umin, umax, norm,  y_start):
    X = np.genfromtxt(file, delimiter=',')
    #x, y, z = X[1:], X[2:], X[3:]
    # show read data
    # df = pd.DataFrame(X)
    # print(df.head(500))

    #visualize noisy image
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(X)
    # o3d.visualization.draw_geometries([pcd])

    l1_norms = np.sum(np.abs(X), axis=1)
    l2_norms = np.linalg.norm(X, axis=1)

    #ver si hay mucho outlier
    #print("Max l1 norms before cleanup: ", np.max(l1_norms))
    #print("Max l2 norms before cleanup: ", np.max(l2_norms))
    #umbral en metros
    umbral_min = umin
    umbral_max = umax

    l1_mask = (l1_norms >= umbral_min) & (l1_norms <= umbral_max)

    l2_mask = (l2_norms >= umbral_min) & (l2_norms <= umbral_max)
    if y_start == 0:
        y_mask =   True
    else:
        y_mask = X[:, 1] < y_start
    l1_mask = l1_mask & y_mask
    l2_mask = l2_mask & y_mask

    if norm == 1:
        filtered_points = X[l1_mask]
    elif norm == 2:
        filtered_points = X[l2_mask]
    else:
        filtered_points = []
    return filtered_points

def compute_curvature(pcd):
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    curvatures = []

    for i in range(len(pcd.points)):
        [_, idx, _] = kdtree.search_knn_vector_3d(pcd.points[i], 30)
        k_neighbors = np.asarray(pcd.points)[idx, :]

        # Compute the covariance matrix
        covariance = np.cov(k_neighbors, rowvar=False)

        # Eigen decomposition of the covariance matrix
        eigenvalues, _ = np.linalg.eigh(covariance)

        # Curvature is given by the ratio of the smallest eigenvalue to the sum of the eigenvalues
        curvature = eigenvalues[0] / np.sum(eigenvalues)
        curvatures.append(curvature)

    return np.array(curvatures)

def visualize(file, umin, umax, norm, y_start, n_sample):
    filtered_points = read_and_clean(file, umin, umax, norm, y_start)
    filtered_points = filtered_points[np.random.choice(filtered_points.shape[0], n_sample, replace=False)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)


    curvatures = compute_curvature(pcd)
    pcd.colors = o3d.utility.Vector3dVector(plt.cm.jet(curvatures / max(curvatures))[:, :3])

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    #pcd.orient_normals_consistent_tangent_plane(100)
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    curvatures = np.asarray(curvatures).reshape(-1, 1)  # Reshape to column vector
    data = np.hstack((points, normals, curvatures))
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)
    aabb_min_bound = aabb.get_min_bound()
    aabb_max_bound = aabb.get_max_bound()
    aabb_extent = aabb_max_bound - aabb_min_bound

    obb = pcd.get_oriented_bounding_box()
    obb.color = (0, 1, 0)
    obb_min_bound = obb.get_min_bound()
    obb_max_bound = obb.get_max_bound()
    obb_extent = obb_max_bound - obb_min_bound

    aabb_width, aabb_height, aabb_depth = aabb_extent
    obb_width, obb_height, obb_depth = obb_extent
    major_axis_horizontal = max(obb_width, obb_height, obb_depth)
    major_axis_vertical = min(obb_width, obb_height, obb_depth)

    bbox_data = np.full((points.shape[0], 6), [aabb_width, aabb_height, aabb_depth, obb_width, obb_height, obb_depth])
    data_with_bbox = np.hstack((data, bbox_data))

    #o3d.visualization.draw_geometries_with_animation_callback([pcd, aabb, obb],rotate_view, window_name=file)

    file = file.replace('csvs', 'output/' + str(n_sample))
    folder_path = os.path.dirname(file)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    np.savetxt(file, data_with_bbox, delimiter=',', header='x,y,z,nx,ny,nz,curvature,AABB_Width,AABB_Height,AABB_Depth,OBB_Width,OBB_Height,OBB_Depth',comments='')


names = ['csvs/lata/lata', 'csvs/casco/casco']
sample_size = [5000, 10000, 15000, 25000]
for j in sample_size:
    umbral_min = 0.3
    umbral_max = 0.6
    for i in range(1, 11):
        visualize(names[0] + str(i) + '.csv', umbral_min, umbral_max, 1, 0.0, j)

    umbral_min = 0.37
    umbral_max = 0.65
    for i in range(1, 11):
        visualize(names[1] + str(i) + '.csv', umbral_min, umbral_max, 2, 0.1, j)
