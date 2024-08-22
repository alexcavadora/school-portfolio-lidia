
import pyrealsense2 as rs
import numpy as np
import open3d as o3d

# Configurar el flujo de datos
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Iniciar la transmisión
pipeline.start(config)

# Streaming loop
frame_count = 0
try:
    
    while (frame_count < 1) == True:
        # Obtener el conjunto de frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        
        if not depth_frame:
            continue
        
        # Convertir a numpy array
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # Crear una nube de puntos
        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        vtx = np.asanyarray(points.get_vertices())
        xyz = np.asanyarray(vtx).view(np.float32).reshape(-1, 3)
        
        # Crear un objeto Open3D PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        
        # Visualizar la nube de puntos
        o3d.visualization.draw_geometries([pcd])
        
        frame_count += 1

finally:
    # Detener la transmisión
    pipeline.stop()

