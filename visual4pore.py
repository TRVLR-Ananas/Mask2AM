import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def visualize_spheres_as_scatter(csv_file, filename="abstract_model.png", max_points=300, alpha=0.7):
    """
    将等效球体可视化为散点图
    参数:
        csv_file: CSV文件路径
        max_points: 最大显示点数量
        point_size_scale: 点大小缩放因子
        alpha: 透明度 (0-1)
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 限制显示数量
    if len(df) > max_points:
        df = df.sort_values('equivalent_radius', ascending=False).head(max_points)

    # 创建图形
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 颜色板
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

    # 准备数据
    x = df['centroid_x'].values
    y = df['centroid_y'].values
    z = df['centroid_z'].values

    # 计算点大小
    radius = df['equivalent_radius'].values

    # 点的大小（散点图的s参数是点面积，所以使用直径的平方）
    # 添加缩放因子控制点的大小
    point_sizes = (radius ** 2) * np.pi * 0.1244

    # 为每个点分配颜色
    point_colors = []
    for i in range(len(df)):
        color = colors[i % len(colors)]
        point_colors.append(color)
    point_colors_array = np.array(point_colors)
    edge_colors_array = point_colors_array * 0.6
    edge_colors = edge_colors_array.tolist()
    # 绘制散点图
    scatter = ax.scatter(
        x, y, z,
        s=point_sizes,  # 点的大小
        c=point_colors,  # 点的颜色
        alpha=alpha,  # 透明度
        edgecolors=edge_colors,  # 边缘颜色
        linewidth=0.5  # 边缘线宽
    )

    # 计算合适的坐标轴范围
    min_x, max_x = x.min(), x.max()
    min_y, max_y = y.min(), y.max()
    min_z, max_z = z.min(), z.max()

    # 考虑点的大小，为坐标轴添加一些边距
    x_range = max_x - min_x
    y_range = max_y - min_y
    z_range = max_z - min_z

    margin = 0.1  # 10%的边距

    ax.set_xlim(min_x - margin * x_range, max_x + margin * x_range)
    ax.set_ylim(min_y - margin * y_range, max_y + margin * y_range)
    ax.set_zlim(min_z - margin * z_range, max_z + margin * z_range)

    # 设置标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_title(f'Fitted Spheres as Scatter Plot ({len(df)} points)\nPoint size represents sphere diameter')

    ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    # plt.show()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def visualize_spheres_as_scatter_rmap(csv_file, filename="abstract_model.png", max_points=300, alpha=0.7):
    """
    将等效球体可视化为散点图
    参数:
        csv_file: CSV文件路径
        max_points: 最大显示点数量
        alpha: 透明度 (0-1)
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 限制显示数量
    if len(df) > max_points:
        df = df.sort_values('equivalent_radius', ascending=False).head(max_points)

    # 创建图形
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 准备数据
    x = df['centroid_x'].values
    y = df['centroid_y'].values
    z = df['centroid_z'].values
    radii = df['equivalent_radius'].values

    # 计算点大小
    point_sizes = (radii ** 2) * np.pi * 0.1244

    # 使用颜色映射替代固定颜色列表
    # 基于半径大小分配颜色
    norm_radii = (radii - radii.min()) / (radii.max() - radii.min())
    cmap = plt.cm.viridis  # 使用viridis颜色映射
    colors = cmap(norm_radii)

    # 计算边缘颜色（填充颜色的0.6倍）
    edge_colors = []
    for color in colors:
        # 只调整RGB分量，保持alpha不变
        if len(color) == 4:  # RGBA
            r, g, b, a = color
            edge_color = (r * 0.6, g * 0.6, b * 0.6, a)
        else:  # RGB
            r, g, b = color[:3]
            edge_color = (r * 0.6, g * 0.6, b * 0.6)
        edge_colors.append(edge_color)

    # 绘制散点图
    scatter = ax.scatter(
        x, y, z,
        s=point_sizes,  # 点的大小
        c=colors,  # 点的颜色（基于颜色映射）
        alpha=alpha,  # 透明度
        edgecolors=edge_colors,  # 边缘颜色
        linewidth=0.5  # 边缘线宽
    )

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=radii.min(),
                                                  vmax=radii.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, aspect=20)
    cbar.set_label('Equivalent Radius')

    # 计算合适的坐标轴范围
    min_x, max_x = x.min(), x.max()
    min_y, max_y = y.min(), y.max()
    min_z, max_z = z.min(), z.max()

    # 考虑点的大小，为坐标轴添加一些边距
    x_range = max_x - min_x
    y_range = max_y - min_y
    z_range = max_z - min_z

    margin = 0.1  # 10%的边距

    ax.set_xlim(min_x - margin * x_range, max_x + margin * x_range)
    ax.set_ylim(min_y - margin * y_range, max_y + margin * y_range)
    ax.set_zlim(min_z - margin * z_range, max_z + margin * z_range)

    # 设置标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_title(
    #     f'Fitted Spheres as Scatter Plot ({len(df)} points)\nPoint size represents sphere cross-sectional area')

    ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    # plt.show()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

# 使用示例
if __name__ == "__main__":
    visualize_spheres_as_scatter(
        "porosity_cluster_info.csv",
        max_points=300000,
        alpha=0.7
    )
    visualize_spheres_as_scatter_rmap(
        "porosity_cluster_info.csv",
        max_points=300000,
        alpha=0.7
    )