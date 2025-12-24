import argparse
import os
import open3d as o3d

from combinevis import visualize_combined
from img2pc import process_multiple_folders

from meshvisual4FX import save_patches_to_csv, visualize_patches_save
from pccluster4pore import cluster_connected_components, save_cluster_info_porosity
from statistics4all import plot_porosity_statistics, plot_plane_features_statistics
from visual4pore import visualize_spheres_as_scatter

parser_path = argparse.ArgumentParser()
parser_path.add_argument("--FX_mask_fold", type=str, required=True,
                         help="path to the fold of fracture masks")
parser_path.add_argument("--pore_mask_fold", type=str, required=True,
                         help="path to the fold of pore masks")
parser_path.add_argument("--pcd_save_fold", type=str, required=True,
                         help="path to the fold of point cloud files")
parser_path.add_argument("--am_save_fold", type=str, required=True,
                         help="path to the fold of abstract model files")
parser_path.add_argument("--statistic_fold", type=str, required=True,
                         help="path to the fold of statistic results")
args = parser_path.parse_args()

print("->Loading binary images... ")
FX_pcd_fold = os.path.join(args.pcd_save_fold, "fracture")
pore_pcd_fold = os.path.join(args.pcd_save_fold, "pore")
subfolder_1 = process_multiple_folders(args.FX_mask_fold, FX_pcd_fold)
subfolder_2 = process_multiple_folders(args.pore_mask_fold, pore_pcd_fold)


print("->Processing point cloud... ")
if subfolder_1 != subfolder_2:
    print("Image folder mismatch")
else:
    for folder in subfolder_1:
        print(f"{folder}")


        FX_pc_path = os.path.join(args.pcd_save_fold, "fracture", f"{folder}.pcd")
        pcd = o3d.io.read_point_cloud(FX_pc_path)
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
        patches_path = os.path.join(args.am_save_fold, f"{folder}", "fracture")
        patches_path_csv = os.path.join(args.am_save_fold,  f"{folder}", "fracture", "patches_info.csv" )
        patches_path_png = os.path.join(args.am_save_fold,  f"{folder}", "fracture", "fracture_abstract_model.png")
        os.makedirs(patches_path, exist_ok=True)
        save_patches_to_csv(patches, patches_path_csv)
        visualize_patches_save(patches, patches_path_png, alpha=0.5)


        pore_pc_path = os.path.join(args.pcd_save_fold, "pore", f"{folder}.pcd")
        points, labels, cluster_info = cluster_connected_components(
            pore_pc_path,
            connectivity=6,
            save_info=True
        )
        pore_am_path = os.path.join(args.am_save_fold, f"{folder}", "pore")
        pore_am_path_csv = os.path.join(args.am_save_fold, f"{folder}", "pore", "porosity_cluster_info.csv")
        pore_am_path_png = os.path.join(args.am_save_fold, f"{folder}", "pore", "pore_abstract_model.png")
        os.makedirs(pore_am_path, exist_ok=True)
        save_cluster_info_porosity(cluster_info, pore_am_path_csv)
        visualize_spheres_as_scatter(
            pore_am_path_csv,
            pore_am_path_png,
            max_points=300000000,
            alpha=0.7
        )

        combine_am_path = os.path.join(args.am_save_fold, f"{folder}", "combine")
        combine_am_path_png = os.path.join(args.am_save_fold, f"{folder}", "combine", "combined_result.png")
        os.makedirs(combine_am_path, exist_ok=True)
        visualize_combined(
            pore_am_path_csv,
            patches,
            filename=combine_am_path_png,
            max_points=300000000
        )

        pore_statistic_path = os.path.join(args.statistic_fold, f"{folder}","pore")
        FX_statistic_path = os.path.join(args.statistic_fold, f"{folder}", "fracture")
        plot_porosity_statistics(pore_am_path_csv,
                                 pore_statistic_path)
        plot_plane_features_statistics(patches_path_csv,
                                       FX_statistic_path)

