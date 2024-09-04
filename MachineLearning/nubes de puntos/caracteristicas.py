import numpy as np
import scipy.spatial as spatial
from sklearn.decomposition import PCA
import time

start = time.time()

def read_and_clean(file, umin, umax, norm, y_start):
    X = np.genfromtxt(file, delimiter=',')
    l1_norms = np.sum(np.abs(X), axis=1)
    l2_norms = np.linalg.norm(X, axis=1)
    l1_mask = (l1_norms >= umin) & (l1_norms <= umax)
    l2_mask = (l2_norms >= umin) & (l2_norms <= umax)
    y_mask = X[:, 1] < y_start if y_start != 0 else np.ones(X.shape[0], dtype=bool)
    l1_mask &= y_mask
    l2_mask &= y_mask

    if norm == 1:
        filtered_points = X[l1_mask]
    elif norm == 2:
        filtered_points = X[l2_mask]
    else:
        filtered_points = np.array([])

    return filtered_points

def compute_curvature(points, k=30):
    kdtree = spatial.KDTree(points)
    curvatures = []

    for i in range(len(points)):
        _, idx = kdtree.query(points[i], k=k)
        neighbors = points[idx]
        covariance = np.cov(neighbors, rowvar=False)
        eigenvalues = np.linalg.eigh(covariance)[0]
        curvature = eigenvalues[0] / np.sum(eigenvalues)
        curvatures.append(curvature)

    return np.array(curvatures)

def estimate_normals(points):
    pca = PCA(n_components=3)
    pca.fit(points)
    normals = pca.components_[2]  # The third component is the normal direction
    return normals

def compute_convex_hull_volume(points):
    hull = spatial.ConvexHull(points)
    return hull.volume

def compute_actual_volume(points):
    """
    Approximate the actual volume enclosed by the point cloud using Delaunay triangulation.
    """
    delaunay = spatial.Delaunay(points)
    simplices = delaunay.simplices
    tetrahedra = points[simplices]

    def tetrahedron_volume(tetra):
        """
        Calculate the volume of a tetrahedron given its vertices.
        """
        a, b, c, d = tetra
        return np.abs(np.dot(a - d, np.cross(b - d, c - d))) / 6.0

    volumes = np.array([tetrahedron_volume(tetra) for tetra in tetrahedra])
    return np.sum(volumes)

def compute_concavity(points):
    convex_hull_volume = compute_convex_hull_volume(points)
    actual_volume = compute_actual_volume(points)
    concavity = convex_hull_volume - actual_volume
    return concavity

def compute_compactness(volume, area):
    return volume / area

def compute_sphericity(volume, area):
    radius = (3 * volume / (4 * np.pi)) ** (1/3)
    sphere_surface_area = 4 * np.pi * radius ** 2
    return area / sphere_surface_area

def compute_roughness(points, k=30):
    kdtree = spatial.KDTree(points)
    roughness_values = []

    for i in range(points.shape[0]):
        _, idx = kdtree.query(points[i], k=k)
        neighbors = points[idx[1:]]
        dists = np.linalg.norm(neighbors - points[i], axis=1)
        roughness_values.append(dists.mean())

    return np.mean(roughness_values)

def visualize(file, umin, umax, norm, y_start, n_sample, target):
    filtered_points = read_and_clean(file, umin, umax, norm, y_start)
    if n_sample > filtered_points.shape[0]:
        n_sample = filtered_points.shape[0]
    filtered_points = filtered_points[np.random.choice(filtered_points.shape[0], n_sample, replace=False)]

    curvatures = compute_curvature(filtered_points)
    mean_curvature = curvatures.mean()

    normals = estimate_normals(filtered_points)
    mean_normals = normals.mean(axis=0)

    obb = spatial.ConvexHull(filtered_points)
    obb_volume = obb.volume
    obb_extent = np.ptp(filtered_points, axis=0)

    major_axis_horizontal = max(obb_extent)
    major_axis_vertical = min(obb_extent)

    concavity = compute_concavity(filtered_points)

    surface_area = obb.area
    compactness = compute_compactness(obb_volume, surface_area)
    sphericity = compute_sphericity(obb_volume, surface_area)
    roughness = compute_roughness(filtered_points)

    single_row_data = np.hstack([
        mean_normals,  # Mean normal vectors
        mean_curvature,             # 1 feature: mean curvature
        obb_extent,                 # 3 features: OBB extents (width, height, depth)
        obb_volume,                 # 1 feature: OBB volume
        major_axis_horizontal,      # 1 feature: OBB major axis horizontal
        major_axis_vertical,        # 1 feature: OBB major axis vertical
        concavity,                  # 1 feature: concavity
        compactness,                # 1 feature: compactness
        sphericity,                 # 1 feature: sphericity
        roughness,                  # 1 feature: roughness
        target
    ])
    return single_row_data.reshape(1, -1)

# Processing data
names = ['200/audifonos_200/audifonos', '200/carro_200/carro','200/bolsa_200/bolsa',
         '200/bota_200/bota','200/molcajete_200/molcajete','200/teclado_200/teclado',
         '200/unicel_l2/unicel','200/zapato_l2/zapato']
sample_size = [5000]
all_data = []

targets = [0, 1, 2, 3, 4, 5]
umbral_min_max = [(0.0, 0.6), (0.0, 10.05), (0.07, 10.05), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]

for target, (umbral_min, umbral_max) in zip(targets, umbral_min_max):
    st = time.time()
    for i in range(1, 201):
        for j in sample_size:
            data = visualize(names[target] + str(i) + '.csv', umbral_min, umbral_max, 1, 0.0, j, target)
            all_data.append(data)
    ft = time.time()
    print(names[target]," tomó ", ft-st, "s.")

combined = np.vstack(all_data)
output_file = 'output/database2.csv'
header = 'Mean_Nx,Mean_Ny,Mean_Nz,Mean_Curvature,OBB_Width,OBB_Height,OBB_Depth,Major_Axis_Horizontal,Major_Axis_Vertical,Convex_Hull_Volume,Surface_Area,Concavity,Target'
np.savetxt(output_file, combined, delimiter=',', header=header, comments='')

end = time.time()
print("Duration = ", end - start)
