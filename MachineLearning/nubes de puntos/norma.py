import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np

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
    # o3d.visualization.draw_geometries([pcd])

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
        covariance = np.cov(k_neighbors, rowvar=False)
        eigenvalues, _ = np.linalg.eigh(covariance)
        curvature = eigenvalues[0] / np.sum(eigenvalues)
        curvatures.append(curvature)

    return np.array(curvatures)

def compute_concavity(pcd):
    convex_hull = pcd.compute_convex_hull()[0]
    convex_hull_volume = convex_hull.get_volume()
    pcd_volume = pcd.get_oriented_bounding_box().volume()
    concavity_value = convex_hull_volume - pcd_volume
    return concavity_value

def compute_compactness(pcd, volume):
    surface_area = pcd.get_surface_area()
    compactness_value = volume / surface_area
    return compactness_value

def compute_sphericity(pcd, volume):
    surface_area = pcd.get_surface_area()
    radius = (3 * volume / (4 * np.pi)) ** (1/3)
    sphere_surface_area = 4 * np.pi * radius ** 2
    sphericity_value = surface_area / sphere_surface_area
    return sphericity_value

def compute_roughness(pcd):
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    points = np.asarray(pcd.points)
    roughness_values = []

    for i in range(points.shape[0]):
        [_, idx, _] = kdtree.search_knn_vector_3d(pcd.points[i], 30)
        neighbors = points[idx[1:], :]
        dists = np.linalg.norm(neighbors - points[i], axis=1)
        roughness_values.append(dists.mean())

    roughness_value = np.mean(roughness_values)
    return roughness_value

file = ""
def visualize(file, umin, umax, norm, y_start, n_sample):
    filtered_points = read_and_clean(file, umin, umax, norm, y_start)
    if n_sample > filtered_points.shape[0]:
        n_sample = filtered_points.shape[0]
    filtered_points = filtered_points[np.random.choice(filtered_points.shape[0], n_sample, replace=False)]
    

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    
    curvatures = compute_curvature(pcd)
    #pcd.colors = o3d.utility.Vector3dVector(plt.cm.jet(curvatures / max(curvatures))[:, :3])
    
    
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=40))
    
   
    normals = np.asarray(pcd.normals)
    curvatures = np.asarray(curvatures).reshape(-1, 1)

   
    mean_normals = normals.mean(axis=0)
    
    
    mean_curvature = curvatures.mean()

    obb = pcd.get_oriented_bounding_box()
    obb_extent = obb.get_extent()

  
    obb_volume = obb_extent[0] * obb_extent[1] * obb_extent[2]

   
    major_axis_horizontal = max(obb_extent)

    major_axis_vertical = min(obb_extent)

    concavity = compute_concavity(pcd)

    compactness = compute_compactness(pcd, obb_volume)

    sphericity = compute_sphericity(pcd)

    roughness = compute_roughness(pcd)


    single_row_data = np.hstack([
        mean_normals,               # 3 features: mean normal vector components (x, y, z)
        mean_curvature,             # 1 feature: mean curvature
        obb_extent,                 # 3 features: OBB extents (width, height, depth)
        obb_volume,                 # 1 feature: OBB volume
        major_axis_horizontal,      # 1 feature: OBB major axis horizontal
        major_axis_vertical,        # 1 feature: OBB major axis vertical
        concavity,                  # 1 feature: concavity
        compactness,                # 1 feature: compactness
        sphericity,                 # 1 feature: sphericity
        roughness                   # 1 feature: roughness
    ])

    single_row_data = single_row_data.reshape(1, -1)
    
    return single_row_data

names = ['200/audifonos_200/audifonos', '200/carro_200/carro','200/bolsa_200/bolsa',
        '200/bota_200/bota','200/molcajete_200/molcajete','200/teclado_200/teclado',
        '200/unicel_l2/unicel','200/zapato_l2/zapato']
sample_size = [15000, 25000]
all_data=[]

target = 0
umbral_min = 0.0
umbral_max = 0.6
for i in range(1, 201):
     for j in sample_size:
        data = visualize(names[0] + str(i) + '.csv', umbral_min, umbral_max, 1, 0.0, j)
        all_data.append(data)

target = 1
umbral_min = 0.0
umbral_max = 10.05
for i in range(1, 201):
     for j in sample_size:
        data = visualize(names[1] + str(i) + '.csv', umbral_min, umbral_max, 1, 0.0, j)
        all_data.append(data)

target = 2
umbral_min = 0.07
umbral_max = 10.05
for i in range(1, 201):
     for j in sample_size:
        data = visualize(names[2] + str(i) + '.csv', umbral_min, umbral_max, 1, 0.0, j)
        all_data.append(data)

target = 3
umbral_min = 0.0
umbral_max = 10.0
for i in range(1, 201):
    for j in sample_size:
        data = visualize(names[3] + str(i) + '.csv', umbral_min, umbral_max, 1, 0.0, j)
        all_data.append(data)

target = 4
umbral_min = 0.0
umbral_max = 10.0
for i in range(1, 201):
    for j in sample_size:
        data = visualize(names[4] + str(i) + '.csv', umbral_min, umbral_max, 1, 0.0, j)
        all_data.append(data)

target = 5
umbral_min = 0.0
umbral_max = 10.0
for i in range(1, 201):
    for j in sample_size:
        data = visualize(names[5] + str(i) + '.csv', umbral_min, umbral_max, 1, 0.0, j)
        all_data.append(data)

# target = 6
# umbral_min = 0.0
# umbral_max = 10.0
# for i in range(1, 201):
#     for j in sample_size:
#         data = visualize(names[6] + str(i) + '.csv', umbral_min, umbral_max, 2, 0.0, j)
#         all_data.append(data)

# target = 7
# umbral_min = 0.0
# umbral_max = 10.0
# for i in range(1, 201):
#     for j in sample_size:
#         data = visualize(names[7] + str(i) + '.csv', umbral_min, umbral_max, 2, 0.0, j)
#         all_data.append(data)

combined = np.vstack(all_data)
output_file = 'output/database.csv'
header = 'Mean_Nx,Mean_Ny,Mean_Nz,Mean_Curvature,AABB_Width,AABB_Height,AABB_Depth,OBB_Width,OBB_Height,OBB_Depth,Major_Axis_Horizontal,Major_Axis_Vertical,Target'
np.savetxt(output_file, combined, delimiter=',', header=header, comments='')
