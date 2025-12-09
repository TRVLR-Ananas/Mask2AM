import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def mesh_patches(patches):
    plane_meshes = []
    for i, patch in enumerate(patches):
        # 获取边界框参数
        center = patch.center
        R = patch.R  # 旋转矩阵
        extent = patch.extent  # 长宽高

        print(f"平面 {i}: 中心={center}, 法向量={R[:, 2]}, 尺寸={extent}")

        # 创建局部坐标系的基向量
        # R的列向量分别是局部坐标系的x、y、z轴
        axis_x = R[:, 0]  # 局部x轴
        axis_y = R[:, 1]  # 局部y轴
        axis_z = R[:, 2]  # 局部z轴(法线方向)

        # 计算四个角点(在局部坐标系中)
        half_w = extent[0] / 2
        half_h = extent[1] / 2

        # 创建一个平面网格
        plane_mesh = o3d.geometry.TriangleMesh()

        # 创建四个顶点
        vertices = [
            center - half_w * axis_x - half_h * axis_y,  # 左下
            center + half_w * axis_x - half_h * axis_y,  # 右下
            center + half_w * axis_x + half_h * axis_y,  # 右上
            center - half_w * axis_x + half_h * axis_y]  # 左上

        plane_mesh.vertices = o3d.utility.Vector3dVector(vertices)

        # 创建三角形
        triangles = [[0, 1, 2], [0, 2, 3]]
        plane_mesh.triangles = o3d.utility.Vector3iVector(triangles)

        # 随机颜色
        color_palette = [
            [1.000, 0.678, 0.678],  # #ffadad
            [1.000, 0.839, 0.647],  # #ffd6a5
            [0.992, 1.000, 0.714],  # #fdffb6
            [0.792, 1.000, 0.749],  # #caffbf
            [0.608, 0.965, 1.000],  # #9bf6ff
            [0.627, 0.769, 1.000],  # #a0c4ff
            [0.741, 0.698, 1.000],  # #bdb2ff
            [1.000, 0.776, 1.000]  # #ffc6ff
        ]
        color_idx = i % len(color_palette)
        color = color_palette[color_idx]
        plane_mesh.paint_uniform_color(color)
        plane_mesh.compute_vertex_normals()
        plane_meshes.append(plane_mesh)
    return plane_meshes

def mesh_patches_with_grid(patches, grid_size=50):
    plane_meshes = []
    for i, patch in enumerate(patches):
        # 获取边界框参数
        center = patch.center
        R = patch.R  # 旋转矩阵
        extent = patch.extent  # 长宽高

        print(f"平面 {i}: 中心={center}, 法向量={R[:, 2]}, 尺寸={extent}")

        # 创建局部坐标系的基向量
        # R的列向量分别是局部坐标系的x、y、z轴
        axis_x = R[:, 0]  # 局部x轴
        axis_y = R[:, 1]  # 局部y轴
        axis_z = R[:, 2]  # 局部z轴(法线方向)

        # 计算四个角点(在局部坐标系中)
        half_w = extent[0] / 2
        half_h = extent[1] / 2

        # 创建一个平面网格
        plane_mesh = o3d.geometry.TriangleMesh()

        # 计算网格数量
        num_w = max(2, int(extent[0] / grid_size))  # 宽度方向网格数，至少2
        num_h = max(2, int(extent[1] / grid_size))  # 高度方向网格数，至少2

        # 创建网格顶点
        vertices = []
        for iy in range(num_h + 1):
            for ix in range(num_w + 1):
                # 计算局部坐标
                local_x = -half_w + (ix / num_w) * extent[0]
                local_y = -half_h + (iy / num_h) * extent[1]

                # 转换为世界坐标
                vertex = center + local_x * axis_x + local_y * axis_y
                vertices.append(vertex)

        # 创建三角形
        triangles = []
        for iy in range(num_h):
            for ix in range(num_w):
                # 当前网格的四个顶点索引
                v00 = iy * (num_w + 1) + ix
                v10 = iy * (num_w + 1) + ix + 1
                v01 = (iy + 1) * (num_w + 1) + ix
                v11 = (iy + 1) * (num_w + 1) + ix + 1

                # 创建两个三角形（构成一个矩形）
                triangles.append([v00, v10, v11])
                triangles.append([v00, v11, v01])

        # 设置顶点和三角形
        plane_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        plane_mesh.triangles = o3d.utility.Vector3iVector(triangles)

        # 创建网格线框（可选）
        wireframe = create_wireframe(i, vertices, num_w, num_h)

        # 随机颜色
        color_palette = [
            [1.000, 0.678, 0.678],  # #ffadad
            [1.000, 0.839, 0.647],  # #ffd6a5
            [0.992, 1.000, 0.714],  # #fdffb6
            [0.792, 1.000, 0.749],  # #caffbf
            [0.608, 0.965, 1.000],  # #9bf6ff
            [0.627, 0.769, 1.000],  # #a0c4ff
            [0.741, 0.698, 1.000],  # #bdb2ff
            [1.000, 0.776, 1.000]  # #ffc6ff
        ]
        color_idx = i % len(color_palette)
        color = color_palette[color_idx]

        # 设置平面颜色
        plane_mesh.paint_uniform_color(color)

        # 如果需要半透明效果，可以启用以下代码
        plane_mesh.compute_vertex_normals()
        plane_mesh.compute_triangle_normals()

        # 可以选择是否显示网格线
        plane_mesh.compute_vertex_normals()
        plane_meshes.append(plane_mesh)

        # 添加网格线（可选，让网格更明显）
        if wireframe is not None:
            plane_meshes.append(wireframe)

    return plane_meshes

def create_wireframe(i, vertices, num_w, num_h):
    lines = []
    # 水平线
    for iy in range(num_h + 1):
        for ix in range(num_w):
            idx1 = iy * (num_w + 1) + ix
            idx2 = iy * (num_w + 1) + ix + 1
            lines.append([idx1, idx2])

    # 垂直线
    for ix in range(num_w + 1):
        for iy in range(num_h):
            idx1 = iy * (num_w + 1) + ix
            idx2 = (iy + 1) * (num_w + 1) + ix
            lines.append([idx1, idx2])

    if lines:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(vertices)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        color_palette = [
            [1.000, 0.678, 0.678],  # #ffadad
            [1.000, 0.839, 0.647],  # #ffd6a5
            [0.992, 1.000, 0.714],  # #fdffb6
            [0.792, 1.000, 0.749],  # #caffbf
            [0.608, 0.965, 1.000],  # #9bf6ff
            [0.627, 0.769, 1.000],  # #a0c4ff
            [0.741, 0.698, 1.000],  # #bdb2ff
            [1.000, 0.776, 1.000]  # #ffc6ff
        ]

        color_idx = i % len(color_palette)
        original_color = color_palette[color_idx]

        darken_factor = 0.6  # 调整这个值可以获得不同的深度
        darker_color = [c * darken_factor for c in original_color]
        line_set.paint_uniform_color(darker_color)

        return line_set

    return None

def plane_visualizer_1(pcd_original, plane_meshes):
    # 可视化
    if plane_meshes:
        width, height = 1000, 1000
        vis1 = o3d.visualization.Visualizer()
        vis1.create_window("原始点云", width, height, 100, 100)
        vis1.add_geometry(pcd_original)
        vis2 = o3d.visualization.Visualizer()
        vis2.create_window("提取的平面", width, height, width+200, 100)
        for mesh in plane_meshes:
            vis2.add_geometry(mesh)

        opt1 = vis1.get_render_option()
        opt1.point_size = 2.0
        opt1.background_color = np.array([1, 1, 1])
        opt2 = vis2.get_render_option()
        opt2.background_color = np.array([1, 1, 1])
        opt2.mesh_show_back_face = True  # 显示背面
        opt2.mesh_color_option = o3d.visualization.MeshColorOption.Color  # 使用顶点颜色
        # opt2.mesh_show_wireframe = True  # 显示线
        opt2.line_width = 2.0
        opt2.light_on = False  # 开启光照

        ctr1 = vis1.get_view_control()
        ctr2 = vis2.get_view_control()
        initial_params = ctr1.convert_to_pinhole_camera_parameters()
        ctr2.convert_from_pinhole_camera_parameters(initial_params)

        try:
            prev_params = initial_params

            while True:
                vis1.poll_events()
                vis1.update_renderer()
                current_params = ctr1.convert_to_pinhole_camera_parameters()
                if not np.allclose(
                        current_params.extrinsic,
                        prev_params.extrinsic,
                        atol=1e-6
                ) or not np.allclose(
                    current_params.intrinsic.intrinsic_matrix,
                    prev_params.intrinsic.intrinsic_matrix,
                    atol=1e-6
                ):
                    ctr2.convert_from_pinhole_camera_parameters(current_params)
                    prev_params = current_params
                vis2.poll_events()
                vis2.update_renderer()

        except KeyboardInterrupt:
            print("程序被用户中断")
        except Exception as e:
            print(f"发生错误: {e}")

        # 关闭窗口
        vis1.destroy_window()
        vis2.destroy_window()

    else:
        print("未检测到平面片")
        o3d.visualization.draw_geometries([pcd_original])


def get_plane_summary(patches):

    print(f"平面总数: {len(patches)}")
    print("\n平面统计信息:")

    # 计算法向量方向统计
    normals = [patch.R[:, 2] for patch in patches]
    normals = np.array(normals)

    print(f"法向量平均值: [{np.mean(normals[:, 0]):.3f}, {np.mean(normals[:, 1]):.3f}, {np.mean(normals[:, 2]):.3f}]")
    print(f"法向量标准差: [{np.std(normals[:, 0]):.3f}, {np.std(normals[:, 1]):.3f}, {np.std(normals[:, 2]):.3f}]")

    # 计算尺寸统计
    extents = [patch.extent for patch in patches]
    extents = np.array(extents)

    print(f"\n平面尺寸统计:")
    print(f"平均尺寸: [{np.mean(extents[:, 0]):.2f}, {np.mean(extents[:, 1]):.2f}, {np.mean(extents[:, 2]):.2f}]")
    print(f"最大尺寸: [{np.max(extents[:, 0]):.2f}, {np.max(extents[:, 1]):.2f}, {np.max(extents[:, 2]):.2f}]")
    print(f"最小尺寸: [{np.min(extents[:, 0]):.2f}, {np.min(extents[:, 1]):.2f}, {np.min(extents[:, 2]):.2f}]")


def visualize_patches_simple(patches, alpha=0.5):

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    all_vertices = []
    colors = [
            [1.000, 0.678, 0.678],  # #ffadad
            [1.000, 0.839, 0.647],  # #ffd6a5
            [0.992, 1.000, 0.714],  # #fdffb6
            [0.792, 1.000, 0.749],  # #caffbf
            [0.608, 0.965, 1.000],  # #9bf6ff
            [0.627, 0.769, 1.000],  # #a0c4ff
            [0.741, 0.698, 1.000],  # #bdb2ff
            [1.000, 0.776, 1.000]  # #ffc6ff
        ]
    for i, patch in enumerate(patches):
        center = patch.center
        R = patch.R
        extent = patch.extent
        half_w = extent[0] / 2
        half_h = extent[1] / 2
        axis_x = R[:, 0]
        axis_y = R[:, 1]
        vertices = np.array([
            center - half_w * axis_x - half_h * axis_y,
            center + half_w * axis_x - half_h * axis_y,
            center + half_w * axis_x + half_h * axis_y,
            center - half_w * axis_x + half_h * axis_y
        ])
        poly = Poly3DCollection([vertices], alpha=alpha, facecolor=colors[i % len(colors)])
        edgecolor = [c * 0.6 for c in colors[i % len(colors)]]
        poly.set_edgecolor(edgecolor)
        poly.set_linewidth(0.5)
        ax.add_collection3d(poly)
        all_vertices.append(vertices)

    if all_vertices:
        all_vertices = np.vstack(all_vertices)
        min_coords = all_vertices.min(axis=0)
        max_coords = all_vertices.max(axis=0)

        ax.set_xlim(min_coords[0], max_coords[0])
        ax.set_ylim(min_coords[1], max_coords[1])
        ax.set_zlim(min_coords[2], max_coords[2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Planes ({len(patches)} planes)', fontsize=14)
    ax.view_init(elev=25, azim=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("->正在加载点云... ")
    pcd = o3d.io.read_point_cloud("C:\\Users\\dell\\PycharmProjects\\Mask2AM\\pcdfold\\S3.pcd")
    print(pcd)
    print("->正在可视化点云")

    pcd_original = o3d.geometry.PointCloud(pcd)
    # pcd_original.paint_uniform_color([0.5, 0.5, 0.5]) # 灰色

    # 估计法线
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
    )

    # 检测平面
    patches = pcd.detect_planar_patches(
        normal_variance_threshold_deg=60,  # 默认值
        coplanarity_deg=75,  # 默认值
        outlier_ratio=0.65,  # 默认值
        min_plane_edge_length=0.0,  # 自动计算
        min_num_points=0,  # 自动计算
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)  # 使用K近邻
    )

    print(f"检测到 {len(patches)} 个平面片")
    mesh_patches_1 = mesh_patches(patches)
    # mesh_patches_1 = mesh_patches_with_grid(patches)
    # plane_visualizer_1(pcd_original, mesh_patches_1)
    visualize_patches_simple(patches, alpha=0.5)
    get_plane_summary(patches)
