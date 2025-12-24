import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import open3d as o3d

def visualize_combined(csv_file, patches, filename="combined_visualization.png",
                       max_points=300, scatter_alpha=0.5, patch_alpha=0.5):
    df = pd.read_csv(csv_file)
    if len(df) > max_points:
        df = df.sort_values('equivalent_radius', ascending=False).head(max_points)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    if len(df) > 0:
        x = df['centroid_x'].values
        y = df['centroid_y'].values
        z = df['centroid_z'].values
        radii = df['equivalent_radius'].values
        point_sizes = (radii ** 2) * np.pi * 0.1244
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
        point_colors = []
        for i in range(len(df)):
            color = colors[i % len(colors)]
            point_colors.append(color)
        point_colors_array = np.array(point_colors)
        edge_colors_array = point_colors_array * 0.6
        edge_colors = edge_colors_array.tolist()
        scatter = ax.scatter(
            x, y, z,
            s=point_sizes,
            c=point_colors,
            alpha=scatter_alpha,
            edgecolors=edge_colors,
            linewidth=0.5
        )
    if patches and len(patches) > 0:
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
            poly = Poly3DCollection([vertices], alpha=patch_alpha,
                                    facecolor=colors[i % len(colors)])
            edgecolor = [c * 0.6 for c in colors[i % len(colors)]]
            poly.set_edgecolor(edgecolor)
            poly.set_linewidth(0.5)
            ax.add_collection3d(poly)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=25, azim=45)
    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    # plt.show()

if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud("C:\\Users\\dell\\PycharmProjects\\Mask2AM\\FINALRESULTS\\pcd\\fracture\\HL1.pcd")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
    )
    patches = pcd.detect_planar_patches(
        normal_variance_threshold_deg=85,
        coplanarity_deg=85,
        outlier_ratio=0.65,
        min_plane_edge_length=50,
        min_num_points=200,
        search_param=o3d.geometry.KDTreeSearchParamKNN(30)
    )
    visualize_combined(
        csv_file="C:\\Users\\dell\\PycharmProjects\\Mask2AM\\FINALRESULTS\\am\\HL1\\pore\\porosity_cluster_info.csv",
        patches=patches,
        filename="combined_result.png",
        max_points=3000
    )