import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import os
from joblib import Parallel, delayed

def read_and_clean(file, umin, umax, norm, y_start):
    X = np.genfromtxt(file, delimiter=',')

    l1_norms = np.sum(np.abs(X), axis=1)
    l2_norms = np.linalg.norm(X, axis=1)

    y_mask = X[:, 1] < y_start if y_start != 0 else np.ones_like(X[:, 1], dtype=bool)

    if norm == 1:
        mask = (l1_norms >= umin) & (l1_norms <= umax) & y_mask
    elif norm == 2:
        mask = (l2_norms >= umin) & (l2_norms <= umax) & y_mask

    return X[mask]

def compute_curvature(pcd):
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    curvatures = []

    for i in range(len(pcd.points)):
        _, idx, _ = kdtree.search_knn_vector_3d(pcd.points[i], 30)
        k_neighbors = np.asarray(pcd.points)[idx, :]

        covariance = np.cov(k_neighbors, rowvar=False)
        eigenvalues, _ = np.linalg.eigh(covariance)

        curvature = eigenvalues[0] / np.sum(eigenvalues)
        curvatures.append(curvature)

    return np.array(curvatures)

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

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    curvatures = np.asarray(curvatures).reshape(-1, 1)

    mean_normals = normals.mean(axis=0)
    mean_curvature = curvatures.mean()

    aabb = pcd.get_axis_aligned_bounding_box()
    obb = pcd.get_oriented_bounding_box()

    aabb_extent = aabb.get_extent()
    obb_extent = obb.get_extent()

    major_axis_horizontal = max(obb_extent)
    major_axis_vertical = min(obb_extent)

    single_row_data = np.hstack([
        mean_normals, mean_curvature,
        aabb_extent, obb_extent,
        major_axis_horizontal, major_axis_vertical
    ])

    return single_row_data.reshape(1, -1)

def process_file(file_prefix, target, umbral_min, umbral_max, norm, y_start, sample_size):
    all_data = []
    for i in range(1, 11):
        for j in sample_size:
            data = visualize(file_prefix + str(i) + '.csv', umbral_min, umbral_max, norm, y_start, j)
            all_data.append(data)
            data = visualize(file_prefix + str(i) + '.csv', umbral_min, umbral_max, norm, y_start, j)
            all_data.append(data)
    combined_data = np.vstack(all_data)
    return combined_data

names = ['csvs/lata/lata', 'csvs/casco/casco', 'csvs/oso/osof',
         'csvs/piedra/piedraf', 'csvs/sombrilla/sombrilla', 'csvs/sombrilla/sombrilla',
         'csvs/unicel_l2/unicel', 'csvs/zapato_l2/zapato']

sample_size = [5000, 10000, 15000, 25000]

params = [
    (names[0], 0, 0.3, 0.6, 1, 0.0),
    (names[1], 1, 0.37, 0.65, 2, 0.1),
    (names[2], 2, 0.37, 0.65, 2, 0.1),
    (names[3], 3, 0.1, 2.0, 2, 0.1),
    (names[4], 4, 0.1, 2.0, 2, 0.1),
    (names[5], 5, 0.1, 2.0, 2, 0.1),
    (names[6], 6, 0.1, 2.0, 2, 0.1),
    (names[7], 7, 0.1, 2.0, 2, 0.1),
]

results = Parallel(n_jobs=-1)(delayed(process_file)(p[0], p[1], p[2], p[3], p[4], p[5], sample_size) for p in params)

combined = np.vstack(results)
output_file = 'output/database.csv'
header = 'Mean_Nx,Mean_Ny,Mean_Nz,Mean_Curvature,AABB_Width,AABB_Height,AABB_Depth,OBB_Width,OBB_Height,OBB_Depth,Major_Axis_Horizontal,Major_Axis_Vertical,Target'
np.savetxt(output_file, combined, delimiter=',', header=header, comments='')
