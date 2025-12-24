import numpy as np
import open3d as o3d
from tqdm import tqdm
import pandas as pd
import os
from scipy.spatial import ConvexHull


def cluster_connected_components(filepath, connectivity=6, save_info=True):
    """
    聚类连通分量并输出每个分量的信息

    参数:
        filepath: 点云文件路径
        connectivity: 连通性(6或26邻域)
        save_info: 是否保存分量信息

    返回:
        points: 点云坐标
        labels: 聚类标签数组
        cluster_info: 聚类信息字典列表
    """

    # 1. 加载点云
    pcd = o3d.io.read_point_cloud(filepath)
    points = np.asarray(pcd.points)
    n_points = len(points)
    # print(f"点数: {n_points:,}")

    # 2. 坐标转为整数（假设是规则网格）
    points_int = np.round(points).astype(int)

    # 3. 创建坐标到索引的映射
    coord_dict = {}
    for i, coord in enumerate(tqdm(points_int, desc="坐标映射")):
        coord_dict[tuple(coord)] = i

    # 4. 定义邻域方向
    if connectivity == 6:
        directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    else:  # 26邻域
        directions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if not (dx == 0 and dy == 0 and dz == 0):
                        directions.append((dx, dy, dz))

    # 5. 并查集实现
    parent = list(range(n_points))
    rank = [0] * n_points

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            if rank[root_x] < rank[root_y]:
                parent[root_x] = root_y
            elif rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
            else:
                parent[root_y] = root_x
                rank[root_x] += 1

    # 6. 连接相邻点
    for i, coord in enumerate(tqdm(points_int, desc="连接点", unit="点")):
        for dx, dy, dz in directions:
            neighbor_coord = (coord[0] + dx, coord[1] + dy, coord[2] + dz)
            if neighbor_coord in coord_dict:
                neighbor_idx = coord_dict[neighbor_coord]
                union(i, neighbor_idx)

    # 7. 生成聚类标签
    root_to_label = {}
    labels = np.zeros(n_points, dtype=int)
    current_label = 0

    for i in tqdm(range(n_points), desc="分配标签", unit="点"):
        root = find(i)
        if root not in root_to_label:
            root_to_label[root] = current_label
            current_label += 1
        labels[i] = root_to_label[root]

    n_clusters = len(root_to_label)

    # 8. 收集聚类信息
    cluster_info = []
    if save_info:
        cluster_info = collect_cluster_info_porosity(points, labels, n_clusters)

    return points, labels, cluster_info


def collect_cluster_info_porosity(points, labels, n_clusters):
    """
    收集孔隙聚类信息（针对CT数据的专用函数）
    """
    cluster_info = []

    # 初始化存储结构
    point_counts = np.zeros(n_clusters, dtype=int)

    # 预分配足够的存储空间
    from collections import defaultdict
    cluster_points_dict = defaultdict(list)

    # 第一遍：收集每个聚类的所有点
    for i, point in enumerate(tqdm(points, desc="收集点", unit="点")):
        label = labels[i]
        point_counts[label] += 1
        cluster_points_dict[label].append(point)

    # 第二遍：计算每个聚类的度量
    for label in tqdm(range(n_clusters), desc="计算度量", unit="聚类"):
        if point_counts[label] == 0:
            continue

        # 获取该聚类的所有点
        cluster_pts = np.array(cluster_points_dict[label])

        # 1. 计算质心（真正的中心点）
        centroid = np.mean(cluster_pts, axis=0)

        # 2. 计算体积（点计数法）
        volume_voxels = point_counts[label]  # 每个点代表一个体素

        # 3. 计算等效球半径
        # 假设孔隙是球体，计算等效球半径
        if volume_voxels > 0:
            # 假设每个体素是单位立方体（体积为1）
            # 球的体积公式: V = (4/3) * π * r³
            # 所以 r = (3V / (4π))^(1/3)
            equivalent_radius = (3 * volume_voxels / (4 * np.pi)) ** (1 / 3)
        else:
            equivalent_radius = 0

        # 4. 计算球形度
        if point_counts[label] >= 4:
            try:
                cluster_points_int = np.round(cluster_pts).astype(int)
                points_set = set([tuple(p) for p in cluster_points_int])

                # 定义6邻域方向（上下左右前后）
                directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

                # 计算表面积（暴露的体素面数）
                surface_area = 0
                for point in points_set:
                    for dx, dy, dz in directions:
                        neighbor = (point[0] + dx, point[1] + dy, point[2] + dz)
                        if neighbor not in points_set:
                            surface_area += 1

                if surface_area > 0:
                    sphericity = (np.pi ** (1 / 3)) * (6 * point_counts[label]) ** (2 / 3) / surface_area
                    # 确保球形度在0-1范围内
                    sphericity = max(0.0, min(1.0, sphericity))
                else:
                    sphericity = 0.0
            except Exception as e:
                sphericity = 0.0

        elif point_counts[label] == 1:
            sphericity = 1  # 单个点可视为完美球形
        elif point_counts[label] == 2:
            sphericity = 0  # 两个点没有球形度
        elif point_counts[label] == 3:
            sphericity = 0  # 三个点也无法形成球体

        # 5. 计算边界框（用于参考）
        bbox_min = np.min(cluster_pts, axis=0)
        bbox_max = np.max(cluster_pts, axis=0)
        bbox_size = bbox_max - bbox_min

        # 收集信息
        info = {
            'label': label,
            'point_count': point_counts[label],
            'volume_voxels': volume_voxels,  # 体素体积
            'centroid_x': centroid[0],
            'centroid_y': centroid[1],
            'centroid_z': centroid[2],
            'equivalent_radius': equivalent_radius,  # 等效球半径
            'sphericity': sphericity,  # 球形度
            'min_x': bbox_min[0],  # 仅作参考
            'min_y': bbox_min[1],
            'min_z': bbox_min[2],
            'max_x': bbox_max[0],
            'max_y': bbox_max[1],
            'max_z': bbox_max[2],
            'bbox_size_x': bbox_size[0],  # 仅作参考
            'bbox_size_y': bbox_size[1],
            'bbox_size_z': bbox_size[2],
        }

        cluster_info.append(info)

    return cluster_info


def save_cluster_info_porosity(cluster_info, output_file="porosity_cluster_info.csv"):
    """
    保存孔隙聚类信息到CSV文件
    """
    if not cluster_info:
        print("没有聚类信息可保存")
        return

    # 转换为DataFrame
    df = pd.DataFrame(cluster_info)

    # 设置列的顺序
    columns_order = [
        'label', 'point_count', 'volume_voxels',
        'centroid_x', 'centroid_y', 'centroid_z',
        'equivalent_radius', 'sphericity',
        'min_x', 'min_y', 'min_z',
        'max_x', 'max_y', 'max_z',
        'bbox_size_x', 'bbox_size_y', 'bbox_size_z'
    ]

    # 确保所有列都存在
    existing_columns = [col for col in columns_order if col in df.columns]
    df = df[existing_columns]

    # 保存到CSV
    df.to_csv(output_file, index=False, float_format='%.6f')
    # print(f"孔隙聚类信息已保存到: {output_file}")

    # # 显示统计信息
    # print(f"\n孔隙聚类统计:")
    # print(f"  总聚类数: {len(cluster_info):,}")
    # print(f"  总孔隙体积(体素): {df['volume_voxels'].sum():,}")
    #
    # if len(cluster_info) > 0:
    #     print(f"\n球形度统计:")
    #     print(f"  平均球形度: {df['sphericity'].mean():.4f}")
    #     print(f"  中位数球形度: {df['sphericity'].median():.4f}")
    #     print(f"  最大球形度: {df['sphericity'].max():.4f}")
    #     print(f"  最小球形度: {df['sphericity'].min():.4f}")
    #
    #     print(f"\n等效半径统计:")
    #     print(f"  平均等效半径: {df['equivalent_radius'].mean():.4f}")
    #     print(f"  中位数等效半径: {df['equivalent_radius'].median():.4f}")
    #     print(f"  最大等效半径: {df['equivalent_radius'].max():.4f}")
    #     print(f"  最小等效半径: {df['equivalent_radius'].min():.4f}")
    #
    #     print(f"\n聚类大小分布:")
    #     size_bins = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    #     for i in range(len(size_bins) - 1):
    #         min_sz = size_bins[i]
    #         max_sz = size_bins[i + 1]
    #         count = ((df['point_count'] >= min_sz) & (df['point_count'] < max_sz)).sum()
    #         if count > 0:
    #             print(f"  {min_sz:4d}-{max_sz:4d} 点: {count:6,d} 个聚类")
    #
    #     large_count = (df['point_count'] >= size_bins[-1]).sum()
    #     if large_count > 0:
    #         print(f"  ≥{size_bins[-1]:4d} 点: {large_count:6,d} 个聚类")

    return df


def print_porosity_summary(cluster_info, n_to_show=20):
    """
    打印孔隙聚类摘要信息
    """
    if not cluster_info:
        print("没有聚类信息")
        return

    # 按大小排序
    sorted_info = sorted(cluster_info, key=lambda x: x['point_count'], reverse=True)

    print(f"\n前 {min(n_to_show, len(sorted_info))} 个最大孔隙:")
    print("=" * 130)
    print(f"{'排名':<5} {'聚类ID':<10} {'点数':<10} {'质心坐标':<35} {'等效半径':<12} {'球形度':<10}")
    print("-" * 130)

    for i, info in enumerate(sorted_info[:n_to_show]):
        centroid_str = f"({info['centroid_x']:.1f}, {info['centroid_y']:.1f}, {info['centroid_z']:.1f})"

        print(f"{i + 1:<5} {info['label']:<10} {info['point_count']:<10,} {centroid_str:<35} "
              f"{info['equivalent_radius']:<12.4f} {info['sphericity']:<10.4f}")


def save_largest_clusters_porosity(points, labels, cluster_info, n_clusters_to_save=1000,
                                   sphericity_threshold=None, save_by_sphericity=False,
                                   output_dir="largest_clusters"):
    """
    保存孔隙聚类为单独的文件，可按球形度筛选

    参数:
        points: 所有点的坐标数组
        labels: 每个点的标签数组
        cluster_info: 聚类信息字典列表
        n_clusters_to_save: 要保存的最大聚类数量
        sphericity_threshold: 球形度阈值，小于该值的聚类才会被保存
        save_by_sphericity: 是否按球形度排序保存（True按球形度升序，False按大小降序）
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 根据球形度阈值筛选聚类
    if sphericity_threshold is not None:
        filtered_info = [info for info in cluster_info if info['sphericity'] < sphericity_threshold]
        print(f"球形度阈值: {sphericity_threshold}")
        print(f"原始聚类数: {len(cluster_info)}")
        print(f"筛选后聚类数: {len(filtered_info)}")
    else:
        filtered_info = cluster_info

    if not filtered_info:
        print("没有满足条件的孔隙聚类")
        return

    # 排序策略：按球形度升序或按大小降序
    if save_by_sphericity:
        # 按球形度升序排序（球形度越低越优先）
        sorted_info = sorted(filtered_info, key=lambda x: x['sphericity'])
        sort_criteria = "球形度升序"
    else:
        # 按大小降序排序（默认）
        sorted_info = sorted(filtered_info, key=lambda x: x['point_count'], reverse=True)
        sort_criteria = "大小降序"

    n_to_save = min(n_clusters_to_save, len(sorted_info))

    # 创建子目录
    if sphericity_threshold is not None:
        subdir_name = f"sphericity_lt_{sphericity_threshold:.2f}"
        if save_by_sphericity:
            subdir_name += "_sorted_by_sphericity"
        output_subdir = os.path.join(output_dir, subdir_name)
    else:
        if save_by_sphericity:
            output_subdir = os.path.join(output_dir, "sorted_by_sphericity")
        else:
            output_subdir = os.path.join(output_dir, "sorted_by_size")

    os.makedirs(output_subdir, exist_ok=True)

    print(f"\n保存条件:")
    print(f"  排序方式: {sort_criteria}")
    if sphericity_threshold is not None:
        print(f"  球形度阈值: < {sphericity_threshold}")
    print(f"  保存数量: {n_to_save} / {len(filtered_info)}")
    print(f"  输出目录: {output_subdir}")

    # 创建汇总信息文件
    summary_file = os.path.join(output_subdir, "clusters_summary.csv")
    summary_data = []

    for i, info in enumerate(tqdm(sorted_info[:n_to_save], desc="保存孔隙文件", unit="个")):
        label = info['label']
        cluster_points = points[labels == label]

        # 创建文件名，包含标签、大小和球形度信息
        filename = f"pore_{label}_size{info['point_count']}_sph{info['sphericity']:.4f}.npy"
        output_file = os.path.join(output_subdir, filename)
        np.save(output_file, cluster_points)

        # 记录汇总信息
        summary_data.append({
            'filename': filename,
            'label': label,
            'point_count': info['point_count'],
            'volume_voxels': info['volume_voxels'],
            'centroid_x': info['centroid_x'],
            'centroid_y': info['centroid_y'],
            'centroid_z': info['centroid_z'],
            'equivalent_radius': info['equivalent_radius'],
            'sphericity': info['sphericity'],
            'bbox_size_x': info['bbox_size_x'],
            'bbox_size_y': info['bbox_size_y'],
            'bbox_size_z': info['bbox_size_z']
        })

    # 保存汇总信息
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_file, index=False, float_format='%.6f')

    # 统计信息
    if summary_data:
        print(f"\n保存完成!")
        print(f"  保存目录: {output_subdir}")
        print(f"  汇总文件: {summary_file}")

        # 显示球形度统计
        sphericities = [info['sphericity'] for info in summary_data]
        print(f"\n保存的孔隙球形度统计:")
        print(f"  平均球形度: {np.mean(sphericities):.4f}")
        print(f"  中位数球形度: {np.median(sphericities):.4f}")
        print(f"  最小球形度: {np.min(sphericities):.4f}")
        print(f"  最大球形度: {np.max(sphericities):.4f}")

        # 显示大小统计
        sizes = [info['point_count'] for info in summary_data]
        print(f"\n保存的孔隙大小统计:")
        print(f"  平均点数: {np.mean(sizes):.1f}")
        print(f"  中位数点数: {np.median(sizes):.1f}")
        print(f"  最小点数: {np.min(sizes)}")
        print(f"  最大点数: {np.max(sizes)}")

        # 显示按球形度分组的数量
        sph_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        print(f"\n球形度分布:")
        for i in range(len(sph_bins) - 1):
            count = len([s for s in sphericities if sph_bins[i] <= s < sph_bins[i + 1]])
            if count > 0:
                print(f"  {sph_bins[i]:.1f}-{sph_bins[i + 1]:.1f}: {count:3d} 个孔隙")


if __name__ == "__main__":
    filepath = "C:\\Users\\dell\\PycharmProjects\\Mask2AM\\pcd\\pore_filled\\S3.pcd"

    # 执行聚类
    points, labels, cluster_info = cluster_connected_components(
        filepath,
        connectivity=6,
        save_info=True
    )

    # 保存孔隙聚类信息
    df = save_cluster_info_porosity(cluster_info, "porosity_cluster_info.csv")

    # 打印摘要
    print_porosity_summary(cluster_info, n_to_show=20)

    # 保存球形度小于0.25的孔隙
    save_largest_clusters_porosity(
        points=points,
        labels=labels,
        cluster_info=cluster_info,
        n_clusters_to_save=100,
        sphericity_threshold=0.4,  # 只保存球形度小于0.5的孔隙
        save_by_sphericity=False,  # 按大小排序
        output_dir="filtered_clusters"
    )

    # 保存球形度最低的100个孔隙
    save_largest_clusters_porosity(
        points=points,
        labels=labels,
        cluster_info=cluster_info,
        n_clusters_to_save=0,
        sphericity_threshold=None,  # 不设阈值，保存所有
        save_by_sphericity=True,  # 按球形度升序排序
        output_dir="low_sphericity_clusters"
    )
