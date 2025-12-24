import numpy as np
import open3d as o3d
from tqdm import tqdm
import pandas as pd
import os

def cluster_connected_components(filepath, connectivity=6, save_info=True):
    pcd = o3d.io.read_point_cloud(filepath)
    points = np.asarray(pcd.points)
    n_points = len(points)
    points_int = np.round(points).astype(int)
    coord_dict = {}
    for i, coord in enumerate(tqdm(points_int, desc="Coordinating mapping")):
        coord_dict[tuple(coord)] = i
    if connectivity == 6:
        directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    else:
        directions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if not (dx == 0 and dy == 0 and dz == 0):
                        directions.append((dx, dy, dz))
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
    for i, coord in enumerate(tqdm(points_int, desc="Connecting points")):
        for dx, dy, dz in directions:
            neighbor_coord = (coord[0] + dx, coord[1] + dy, coord[2] + dz)
            if neighbor_coord in coord_dict:
                neighbor_idx = coord_dict[neighbor_coord]
                union(i, neighbor_idx)
    root_to_label = {}
    labels = np.zeros(n_points, dtype=int)
    current_label = 0
    for i in tqdm(range(n_points), desc="Assigning labels"):
        root = find(i)
        if root not in root_to_label:
            root_to_label[root] = current_label
            current_label += 1
        labels[i] = root_to_label[root]
    n_clusters = len(root_to_label)
    cluster_info = []
    if save_info:
        cluster_info = collect_cluster_info_porosity(points, labels, n_clusters)
    return points, labels, cluster_info


def collect_cluster_info_porosity(points, labels, n_clusters):
    cluster_info = []
    point_counts = np.zeros(n_clusters, dtype=int)
    from collections import defaultdict
    cluster_points_dict = defaultdict(list)
    for i, point in enumerate(tqdm(points, desc="Collecting points")):
        label = labels[i]
        point_counts[label] += 1
        cluster_points_dict[label].append(point)
    for label in tqdm(range(n_clusters), desc="Calculating metrics"):
        if point_counts[label] == 0:
            continue
        cluster_pts = np.array(cluster_points_dict[label])
        centroid = np.mean(cluster_pts, axis=0)
        volume_voxels = point_counts[label]
        if volume_voxels > 0:
            equivalent_radius = (3 * volume_voxels / (4 * np.pi)) ** (1 / 3)
        else:
            equivalent_radius = 0
        if point_counts[label] >= 4:
            try:
                cluster_points_int = np.round(cluster_pts).astype(int)
                points_set = set([tuple(p) for p in cluster_points_int])
                directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
                surface_area = 0
                for point in points_set:
                    for dx, dy, dz in directions:
                        neighbor = (point[0] + dx, point[1] + dy, point[2] + dz)
                        if neighbor not in points_set:
                            surface_area += 1
                if surface_area > 0:
                    sphericity = (np.pi ** (1 / 3)) * (6 * point_counts[label]) ** (2 / 3) / surface_area
                    sphericity = max(0.0, min(1.0, sphericity))
                else:
                    sphericity = 0.0
            except Exception as e:
                sphericity = 0.0
        elif point_counts[label] == 1:
            sphericity = 1
        elif point_counts[label] == 2:
            sphericity = 0
        elif point_counts[label] == 3:
            sphericity = 0
        bbox_min = np.min(cluster_pts, axis=0)
        bbox_max = np.max(cluster_pts, axis=0)
        bbox_size = bbox_max - bbox_min
        info = {
            'label': label,
            'point_count': point_counts[label],
            'volume_voxels': volume_voxels,
            'centroid_x': centroid[0],
            'centroid_y': centroid[1],
            'centroid_z': centroid[2],
            'equivalent_radius': equivalent_radius,
            'sphericity': sphericity,
            'min_x': bbox_min[0],
            'min_y': bbox_min[1],
            'min_z': bbox_min[2],
            'max_x': bbox_max[0],
            'max_y': bbox_max[1],
            'max_z': bbox_max[2],
            'bbox_size_x': bbox_size[0],
            'bbox_size_y': bbox_size[1],
            'bbox_size_z': bbox_size[2],
        }

        cluster_info.append(info)
    return cluster_info

def save_cluster_info_porosity(cluster_info, output_file="porosity_cluster_info.csv"):
    if not cluster_info:
        print("No clustering information to save")
        return None
    df = pd.DataFrame(cluster_info)
    columns_order = [
        'label', 'point_count', 'volume_voxels',
        'centroid_x', 'centroid_y', 'centroid_z',
        'equivalent_radius', 'sphericity',
        'min_x', 'min_y', 'min_z',
        'max_x', 'max_y', 'max_z',
        'bbox_size_x', 'bbox_size_y', 'bbox_size_z'
    ]
    existing_columns = [col for col in columns_order if col in df.columns]
    df = df[existing_columns]
    df.to_csv(output_file, index=False, float_format='%.6f')
    return df

if __name__ == "__main__":
    filepath = "C:\\Users\\dell\\PycharmProjects\\Mask2AM\\pcd\\pore_filled\\S3.pcd"
    points, labels, cluster_info = cluster_connected_components(
        filepath,
        connectivity=6,
        save_info=True
    )

    df = save_cluster_info_porosity(cluster_info, "porosity_cluster_info.csv")

