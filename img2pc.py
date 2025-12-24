import numpy as np
import open3d as o3d
import cv2
import os

def binary_images_to_pointcloud_simple(image_folder, output_pcd="output.pcd"):

    # 读取图像序列
    image_files = sorted([f for f in os.listdir(image_folder)
                          if f.endswith(('.png', '.jpg', '.bmp', '.tif'))])

    points = []

    for z_idx, img_file in enumerate(image_files):
        # 读取图像
        img = cv2.imread(os.path.join(image_folder, img_file), cv2.IMREAD_GRAYSCALE)

        # 确保是二值图像
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # 获取白色像素的坐标
        y_coords, x_coords = np.where(binary > 0)

        # 添加到点列表
        for x, y in zip(x_coords, y_coords):
            points.append([x, y, z_idx])

    # 转换为numpy数组
    points = np.array(points)

    # print(points)

    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 保存点云
    o3d.io.write_point_cloud(output_pcd, pcd)

    # print(f"生成 {len(points)} 个点")
    return pcd


def process_multiple_folders(root_dir, output_dir="output_pcds"):
    """
    处理多个文件夹下的图片

    参数:
    root_dir: 包含多个子文件夹的根目录
    output_dir: 保存PCD文件的输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有子文件夹
    subfolders = [f for f in os.listdir(root_dir)
                  if os.path.isdir(os.path.join(root_dir, f))]

    if not subfolders:
        print(f"在 {root_dir} 中未找到子文件夹")
        return

    # 处理每个子文件夹
    for folder in subfolders:
        folder_path = os.path.join(root_dir, folder)

        # 检查文件夹中是否有图片文件
        image_files = [f for f in os.listdir(folder_path)
                       if f.endswith(('.png', '.jpg', '.bmp', '.tif'))]

        if not image_files:
            print(f"文件夹 {folder} 中没有图片文件，跳过")
            continue

        # 生成输出文件名
        output_filename = f"{folder}.pcd"
        output_path = os.path.join(output_dir, output_filename)

        # print(f"处理文件夹: {folder} ({len(image_files)} 张图片)")

        try:
            # 调用原始函数处理
            pcd = binary_images_to_pointcloud_simple(folder_path, output_path)
            # print(f"✓ 完成: {folder} -> {output_filename}\n")
        except Exception as e:
            print(f"✗ 处理 {folder} 时出错: {e}\n")

    return subfolders


if __name__ == "__main__":
    path_1 = "C:\\Users\\dell\\PycharmProjects\\Mask2AM\\img\\pore_edge\\"
    path_2 = "C:\\Users\\dell\\PycharmProjects\\Mask2AM\\pcd\\pore_edge\\"
    process_multiple_folders(path_1, path_2)