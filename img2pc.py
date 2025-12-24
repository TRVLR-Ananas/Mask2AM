import numpy as np
import open3d as o3d
import cv2
import os

def binary_images_to_pointcloud_simple(image_folder, output_pcd="output.pcd"):
    image_files = sorted([f for f in os.listdir(image_folder)
                          if f.endswith(('.png', '.jpg', '.bmp', '.tif'))])
    points = []
    for z_idx, img_file in enumerate(image_files):
        img = cv2.imread(os.path.join(image_folder, img_file), cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        y_coords, x_coords = np.where(binary > 0)
        for x, y in zip(x_coords, y_coords):
            points.append([x, y, z_idx])
    points = np.array(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(output_pcd, pcd)
    return pcd

def process_multiple_folders(root_dir, output_dir="output_pcds"):
    os.makedirs(output_dir, exist_ok=True)
    subfolders = [f for f in os.listdir(root_dir)
                  if os.path.isdir(os.path.join(root_dir, f))]
    if not subfolders:
        print(f"There is no subfolder in {root_dir}")
        return
    for folder in subfolders:
        folder_path = os.path.join(root_dir, folder)
        image_files = [f for f in os.listdir(folder_path)
                       if f.endswith(('.png', '.jpg', '.bmp', '.tif'))]
        if not image_files:
            print(f"There is no image in {folder}")
            continue
        output_filename = f"{folder}.pcd"
        output_path = os.path.join(output_dir, output_filename)
        try:
            pcd = binary_images_to_pointcloud_simple(folder_path, output_path)
        except Exception as e:
            print(f"✗ Wrong with {folder} : {e}\n")
    return subfolders

if __name__ == "__main__":
    path_1 = "C:\\Users\\dell\\PycharmProjects\\Mask2AM\\img\\pore_edge\\"
    path_2 = "C:\\Users\\dell\\PycharmProjects\\Mask2AM\\pcd\\pore_edge\\"
    process_multiple_folders(path_1, path_2)