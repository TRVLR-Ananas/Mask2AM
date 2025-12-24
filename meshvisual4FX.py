import csv
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def mesh_patches(patches):
    plane_meshes = []
    for i, patch in enumerate(patches):
        center = patch.center
        R = patch.R
        extent = patch.extent
        axis_x = R[:, 0]
        axis_y = R[:, 1]
        axis_z = R[:, 2]
        half_w = extent[0] / 2
        half_h = extent[1] / 2
        plane_mesh = o3d.geometry.TriangleMesh()
        vertices = [
            center - half_w * axis_x - half_h * axis_y,
            center + half_w * axis_x - half_h * axis_y,
            center + half_w * axis_x + half_h * axis_y,
            center - half_w * axis_x + half_h * axis_y]
        plane_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        triangles = [[0, 1, 2], [0, 2, 3]]
        plane_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        color_palette = [
            [1.000, 0.678, 0.678],
            [1.000, 0.839, 0.647],
            [0.992, 1.000, 0.714],
            [0.792, 1.000, 0.749],
            [0.608, 0.965, 1.000],
            [0.627, 0.769, 1.000],
            [0.741, 0.698, 1.000],
            [1.000, 0.776, 1.000]
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
        center = patch.center
        R = patch.R
        extent = patch.extent
        axis_x = R[:, 0]
        axis_y = R[:, 1]
        axis_z = R[:, 2]
        half_w = extent[0] / 2
        half_h = extent[1] / 2
        plane_mesh = o3d.geometry.TriangleMesh()
        num_w = max(2, int(extent[0] / grid_size))
        num_h = max(2, int(extent[1] / grid_size))
        vertices = []
        for iy in range(num_h + 1):
            for ix in range(num_w + 1):
                local_x = -half_w + (ix / num_w) * extent[0]
                local_y = -half_h + (iy / num_h) * extent[1]
                vertex = center + local_x * axis_x + local_y * axis_y
                vertices.append(vertex)
        triangles = []
        for iy in range(num_h):
            for ix in range(num_w):
                v00 = iy * (num_w + 1) + ix
                v10 = iy * (num_w + 1) + ix + 1
                v01 = (iy + 1) * (num_w + 1) + ix
                v11 = (iy + 1) * (num_w + 1) + ix + 1
                triangles.append([v00, v10, v11])
                triangles.append([v00, v11, v01])
        plane_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        plane_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        wireframe = create_wireframe(i, vertices, num_w, num_h)
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
        plane_mesh.compute_triangle_normals()
        plane_mesh.compute_vertex_normals()
        plane_meshes.append(plane_mesh)
        if wireframe is not None:
            plane_meshes.append(wireframe)
    return plane_meshes

def create_wireframe(i, vertices, num_w, num_h):
    lines = []
    for iy in range(num_h + 1):
        for ix in range(num_w):
            idx1 = iy * (num_w + 1) + ix
            idx2 = iy * (num_w + 1) + ix + 1
            lines.append([idx1, idx2])
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
            [1.000, 0.678, 0.678],
            [1.000, 0.839, 0.647],
            [0.992, 1.000, 0.714],
            [0.792, 1.000, 0.749],
            [0.608, 0.965, 1.000],
            [0.627, 0.769, 1.000],
            [0.741, 0.698, 1.000],
            [1.000, 0.776, 1.000]
        ]
        color_idx = i % len(color_palette)
        original_color = color_palette[color_idx]
        darken_factor = 0.6
        darker_color = [c * darken_factor for c in original_color]
        line_set.paint_uniform_color(darker_color)
        return line_set
    return None

def plane_visualizer_1(pcd_original, plane_meshes):
    if plane_meshes:
        width, height = 1000, 1000
        vis1 = o3d.visualization.Visualizer()
        vis1.create_window("Origin", width, height, 100, 100)
        vis1.add_geometry(pcd_original)
        vis2 = o3d.visualization.Visualizer()
        vis2.create_window("Planes", width, height, width+200, 100)
        for mesh in plane_meshes:
            vis2.add_geometry(mesh)
        opt1 = vis1.get_render_option()
        opt1.point_size = 2.0
        opt1.background_color = np.array([1, 1, 1])
        opt2 = vis2.get_render_option()
        opt2.background_color = np.array([1, 1, 1])
        opt2.mesh_show_back_face = True
        opt2.mesh_color_option = o3d.visualization.MeshColorOption.Color
        opt2.line_width = 2.0
        opt2.light_on = False
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
            print("Program interrupted by user")
        except Exception as e:
            print(f"An error occurred: {e}")
        vis1.destroy_window()
        vis2.destroy_window()
    else:
        print("No planar patches detected")
        o3d.visualization.draw_geometries([pcd_original])

def visualize_patches_save(patches, filename="abstract_model.png", alpha=0.5):
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
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_patches_to_csv(patches, filename="patches_info.csv"):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Plane_ID',
            'Center_X', 'Center_Y', 'Center_Z',
            'Normal_X', 'Normal_Y', 'Normal_Z',
            'Length', 'Width'
        ])
        for i, patch in enumerate(patches):
            center = patch.center
            R = patch.R
            extent = patch.extent
            row_data = [
                i,
                center[0], center[1], center[2],
                R[0, 2], R[1, 2], R[2, 2],
                extent[0], extent[1]
            ]
            writer.writerow(row_data)

if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud("C:\\Users\\dell\\PycharmProjects\\Mask2AM\\pcd\\FX\\S3.pcd")
    print(pcd)
    pcd_original = o3d.geometry.PointCloud(pcd)
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
    save_patches_to_csv(patches, "patches_info.csv")
    print(f" {len(patches)} ")
    # mesh_patches_1 = mesh_patches(patches)
    mesh_patches_1 = mesh_patches_with_grid(patches)
    plane_visualizer_1(pcd_original, mesh_patches_1)
    # visualize_patches_save(patches, alpha=0.5)

