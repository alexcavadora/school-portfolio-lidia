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

    #visualize noisy image
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(X)
    # o3d.visualization.draw_geometries([pcd])

    l1_norms = np.sum(np.abs(X), axis=1)
    l2_norms = np.linalg.norm(X, axis=1)
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
file = ""
def visualize(file, umin, umax, norm, y_start, n_sample):
    filtered_points = read_and_clean(file, umin, umax, norm, y_start)
    if n_sample > filtered_points.shape[0]:
        n_sample = filtered_points.shape[0]
    filtered_points = filtered_points[np.random.choice(filtered_points.shape[0], n_sample, replace=False)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)


    curvatures = compute_curvature(pcd)
    pcd.colors = o3d.utility.Vector3dVector(plt.cm.jet(curvatures / max(curvatures))[:, :3])

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    #pcd.orient_normals_consistent_tangent_plane(100)
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    curvatures = np.asarray(curvatures).reshape(-1, 1)

    mean_normals = normals.mean(axis=0)
    mean_curvature = curvatures.mean()

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

    single_row_data = np.hstack([
        mean_normals,
        mean_curvature,
        aabb_width, aabb_height, aabb_depth,
        obb_width, obb_height, obb_depth,
        major_axis_horizontal, major_axis_vertical, target
    ])

    single_row_data = single_row_data.reshape(1, -1)
    return single_row_data

names = ['csvs/lata/lata', 'csvs/casco/casco','csvs/oso/osof',
        'csvs/piedra/piedraf','csvs/sombrilla/sombrilla','csvs/audifonos/audi',
        'csvs/unicel_l2/unicel','csvs/zapato_l2/zapato']
sample_size = [5000, 10000, 15000, 25000]
all_data=[]

target = 0

umbral_min = 0.3
umbral_max = 0.6
for i in range(1, 11):
     for j in sample_size:
        data = visualize(names[0] + str(i) + '.csv', umbral_min, umbral_max, 1, 0.0, j)
        all_data.append(data)
        data = visualize(names[0] + str(i) + '.csv', umbral_min, umbral_max, 1, 0.0, j)
        all_data.append(data)
target = 1
umbral_min = 0.37
umbral_max = 0.65
for i in range(1, 11):
     for j in sample_size:
        data = visualize(names[1] + str(i) + '.csv', umbral_min, umbral_max, 1, 0.0, j)
        all_data.append(data)
        data = visualize(names[1] + str(i) + '.csv', umbral_min, umbral_max, 1, 0.0, j)
        all_data.append(data)
target = 2
umbral_min = 0.37
umbral_max = 0.65
for i in range(1, 11):
     for j in sample_size:
        data = visualize(names[2] + str(i) + '.csv', umbral_min, umbral_max, 1, 0.0, j)
        all_data.append(data)
        data = visualize(names[2] + str(i) + '.csv', umbral_min, umbral_max, 1, 0.0, j)
        all_data.append(data)
target = 3
umbral_min = 0.1
umbral_max = 2.0
for i in range(1, 11):
    for j in sample_size:
        data = visualize(names[3] + str(i) + '.csv', umbral_min, umbral_max, 1, 0.0, j)
        all_data.append(data)
        data = visualize(names[3] + str(i) + '.csv', umbral_min, umbral_max, 1, 0.0, j)
        all_data.append(data)
target = 4
umbral_min = 0.1
umbral_max = 2.0
for i in range(1, 11):
    for j in sample_size:
        data = visualize(names[4] + str(i) + '.csv', umbral_min, umbral_max, 1, 0.0, j)
        all_data.append(data)
        data = visualize(names[4] + str(i) + '.csv', umbral_min, umbral_max, 1, 0.0, j)
        all_data.append(data)
target = 5
umbral_min = 0.1
umbral_max = 2.0
for i in range(1, 11):
    for j in sample_size:
        data = visualize(names[5] + str(i) + '.csv', umbral_min, umbral_max, 1, 0.0, j)
        all_data.append(data)
        data = visualize(names[5] + str(i) + '.csv', umbral_min, umbral_max, 1, 0.0, j)
        all_data.append(data)
target = 6
umbral_min = 0.1
umbral_max = 2.0
for i in range(1, 11):
    for j in sample_size:
        data = visualize(names[6] + str(i) + '.csv', umbral_min, umbral_max, 2, 0.0, j)
        all_data.append(data)
        data = visualize(names[6] + str(i) + '.csv', umbral_min, umbral_max, 2, 0.0, j)
        all_data.append(data)

target = 7
umbral_min = 0.1
umbral_max = 2.0
for i in range(1, 11):
    for j in sample_size:
        data = visualize(names[7] + str(i) + '.csv', umbral_min, umbral_max, 2, 0.0, j)
        all_data.append(data)
        data = visualize(names[7] + str(i) + '.csv', umbral_min, umbral_max, 2, 0.0, j)
        all_data.append(data)

combined = np.vstack(all_data)
output_file = 'output/database.csv'
header = 'Mean_Nx,Mean_Ny,Mean_Nz,Mean_Curvature,AABB_Width,AABB_Height,AABB_Depth,OBB_Width,OBB_Height,OBB_Depth,Major_Axis_Horizontal,Major_Axis_Vertical,Target'
np.savetxt(output_file, combined, delimiter=',', header=header, comments='')
