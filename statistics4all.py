import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_porosity_statistics(csv_file_path, output_dir="porosity_statistics"):
    """
    读取孔隙聚类信息CSV文件并绘制独立的统计图

    参数:
        csv_file_path: CSV文件路径
        output_dir: 输出图像目录
    """

    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取数据
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return None

    # 2. 创建独立的图形

    # 图1: 等效半径分布直方图
    if 'equivalent_radius' in df.columns:
        fig1 = plt.figure(figsize=(10, 8))
        plt.hist(df['equivalent_radius'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        plt.xlabel('Equivalent Radius', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title('Equivalent Radius Distribution', fontsize=16, fontweight='bold')
        plt.axvline(df['equivalent_radius'].mean(), color='red', linestyle='--',
                    label=f'Mean: {df["equivalent_radius"].mean():.3f}')
        plt.axvline(df['equivalent_radius'].median(), color='blue', linestyle=':',
                    label=f'Median: {df["equivalent_radius"].median():.3f}')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig1.savefig(f'{output_dir}/equivalent_radius_histogram.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)

    # 图2: 等效半径箱线图
    if 'equivalent_radius' in df.columns:
        fig2 = plt.figure(figsize=(8, 10))
        plt.boxplot(df['equivalent_radius'], vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightcoral'))
        plt.ylabel('Equivalent Radius', fontsize=14)
        plt.title('Equivalent Radius Box Plot', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig2.savefig(f'{output_dir}/equivalent_radius_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)

    # 图3: 等效半径与点数量关系散点图
    if 'equivalent_radius' in df.columns and 'point_count' in df.columns:
        fig3 = plt.figure(figsize=(10, 8))
        plt.scatter(df['point_count'], df['equivalent_radius'],
                    alpha=0.6, s=20, color='darkorange')
        plt.xlabel('Point Count', fontsize=14)
        plt.ylabel('Equivalent Radius', fontsize=14)
        plt.title('Equivalent Radius vs Point Count', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.tight_layout()
        fig3.savefig(f'{output_dir}/equivalent_radius_vs_point_count.png', dpi=300, bbox_inches='tight')
        plt.close(fig3)

    # 图4: 等效半径累积分布函数
    if 'equivalent_radius' in df.columns:
        fig4 = plt.figure(figsize=(10, 8))
        sorted_radius = np.sort(df['equivalent_radius'])
        y_vals = np.arange(1, len(sorted_radius) + 1) / len(sorted_radius)
        plt.plot(sorted_radius, y_vals, 'b-', linewidth=2)
        plt.xlabel('Equivalent Radius', fontsize=14)
        plt.ylabel('Cumulative Probability', fontsize=14)
        plt.title('Equivalent Radius CDF', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # 标记关键分位数
        for quantile in [0.25, 0.5, 0.75, 0.9]:
            idx = int(quantile * len(sorted_radius))
            if idx < len(sorted_radius):
                plt.axvline(sorted_radius[idx], color='red', linestyle='--', alpha=0.5)
                plt.text(sorted_radius[idx] * 1.1, quantile - 0.05,
                         f'{quantile * 100:.0f}%: {sorted_radius[idx]:.3f}',
                         fontsize=9, color='red')

        plt.tight_layout()
        fig4.savefig(f'{output_dir}/equivalent_radius_cdf.png', dpi=300, bbox_inches='tight')
        plt.close(fig4)

    # 图5: 三维质心分布
    if all(col in df.columns for col in ['centroid_x', 'centroid_y', 'centroid_z']):
        fig5 = plt.figure(figsize=(12, 10))
        ax5 = fig5.add_subplot(111, projection='3d')
        if 'equivalent_radius' in df.columns:
            scatter5 = ax5.scatter(df['centroid_x'], df['centroid_y'], df['centroid_z'],
                                   c=df['equivalent_radius'], alpha=0.6, s=15, cmap='viridis')
            plt.colorbar(scatter5, ax=ax5, shrink=0.8, label='Equivalent Radius')
        else:
            ax5.scatter(df['centroid_x'], df['centroid_y'], df['centroid_z'],
                        alpha=0.6, s=15, color='blue')

        ax5.set_xlabel('X Coordinate', fontsize=14, labelpad=10)
        ax5.set_ylabel('Y Coordinate', fontsize=14, labelpad=10)
        ax5.set_zlabel('Z Coordinate', fontsize=14, labelpad=10)
        ax5.set_title('3D Centroid Distribution', fontsize=16, fontweight='bold', pad=20)
        ax5.view_init(elev=20, azim=45)
        plt.tight_layout()
        fig5.savefig(f'{output_dir}/3d_centroid_distribution.png', dpi=300, bbox_inches='tight')
        plt.close(fig5)

    # 图6: X-Y平面投影
    if all(col in df.columns for col in ['centroid_x', 'centroid_y']):
        fig6 = plt.figure(figsize=(10, 8))
        ax6 = fig6.add_subplot(111)
        if 'equivalent_radius' in df.columns:
            scatter6 = ax6.scatter(df['centroid_x'], df['centroid_y'],
                                   c=df['equivalent_radius'], alpha=0.6, s=15, cmap='viridis')
            plt.colorbar(scatter6, ax=ax6, shrink=0.8, label='Equivalent Radius')
        else:
            ax6.scatter(df['centroid_x'], df['centroid_y'], alpha=0.6, s=15, color='blue')

        ax6.set_xlabel('X Coordinate', fontsize=14)
        ax6.set_ylabel('Y Coordinate', fontsize=14)
        ax6.set_title('Centroid X-Y Projection', fontsize=16, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        plt.tight_layout()
        fig6.savefig(f'{output_dir}/centroid_xy_projection.png', dpi=300, bbox_inches='tight')
        plt.close(fig6)

    # 图7: X-Z平面投影
    if all(col in df.columns for col in ['centroid_x', 'centroid_z']):
        fig7 = plt.figure(figsize=(10, 8))
        ax7 = fig7.add_subplot(111)
        if 'equivalent_radius' in df.columns:
            scatter7 = ax7.scatter(df['centroid_x'], df['centroid_z'],
                                   c=df['equivalent_radius'], alpha=0.6, s=15, cmap='viridis')
            plt.colorbar(scatter7, ax=ax7, shrink=0.8, label='Equivalent Radius')
        else:
            ax7.scatter(df['centroid_x'], df['centroid_z'], alpha=0.6, s=15, color='blue')

        ax7.set_xlabel('X Coordinate', fontsize=14)
        ax7.set_ylabel('Z Coordinate', fontsize=14)
        ax7.set_title('Centroid X-Z Projection', fontsize=16, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        plt.tight_layout()
        fig7.savefig(f'{output_dir}/centroid_xz_projection.png', dpi=300, bbox_inches='tight')
        plt.close(fig7)

    # 图8: Y-Z平面投影
    if all(col in df.columns for col in ['centroid_y', 'centroid_z']):
        fig8 = plt.figure(figsize=(10, 8))
        ax8 = fig8.add_subplot(111)
        if 'point_count' in df.columns:
            scatter8 = ax8.scatter(df['centroid_y'], df['centroid_z'],
                                   c=df['equivalent_radius'], alpha=0.6, s=15, cmap='viridis')
            plt.colorbar(scatter8, ax=ax8, shrink=0.8, label='Equivalent Radius')
        else:
            ax8.scatter(df['centroid_y'], df['centroid_z'], alpha=0.6, s=15, color='blue')

        ax8.set_xlabel('Y Coordinate', fontsize=14)
        ax8.set_ylabel('Z Coordinate', fontsize=14)
        ax8.set_title('Centroid Y-Z Projection', fontsize=16, fontweight='bold')
        ax8.grid(True, alpha=0.3)
        plt.tight_layout()
        fig8.savefig(f'{output_dir}/centroid_yz_projection.png', dpi=300, bbox_inches='tight')
        plt.close(fig8)

    # 图9: 质心X坐标分布直方图
    if 'centroid_x' in df.columns:
        fig9 = plt.figure(figsize=(10, 8))
        plt.hist(df['centroid_x'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        plt.xlabel('X Coordinate', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title('Centroid X Distribution', fontsize=16, fontweight='bold')
        plt.axvline(df['centroid_x'].mean(), color='red', linestyle='--',
                    label=f'Mean: {df["centroid_x"].mean():.1f}')
        plt.axvline(df['centroid_x'].median(), color='blue', linestyle=':',
                    label=f'Median: {df["centroid_x"].median():.1f}')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig9.savefig(f'{output_dir}/centroid_x_distribution.png', dpi=300, bbox_inches='tight')
        plt.close(fig9)

    # 图10: 质心Y坐标分布直方图
    if 'centroid_y' in df.columns:
        fig10 = plt.figure(figsize=(10, 8))
        plt.hist(df['centroid_y'], bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
        plt.xlabel('Y Coordinate', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title('Centroid Y Distribution', fontsize=16, fontweight='bold')
        plt.axvline(df['centroid_y'].mean(), color='red', linestyle='--',
                    label=f'Mean: {df["centroid_y"].mean():.1f}')
        plt.axvline(df['centroid_y'].median(), color='blue', linestyle=':',
                    label=f'Median: {df["centroid_y"].median():.1f}')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig10.savefig(f'{output_dir}/centroid_y_distribution.png', dpi=300, bbox_inches='tight')
        plt.close(fig10)

    # 图11: 质心Z坐标分布直方图
    if 'centroid_z' in df.columns:
        fig11 = plt.figure(figsize=(10, 8))
        plt.hist(df['centroid_z'], bins=30, edgecolor='black', alpha=0.7, color='violet')
        plt.xlabel('Z Coordinate', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title('Centroid Z Distribution', fontsize=16, fontweight='bold')
        plt.axvline(df['centroid_z'].mean(), color='red', linestyle='--',
                    label=f'Mean: {df["centroid_z"].mean():.1f}')
        plt.axvline(df['centroid_z'].median(), color='blue', linestyle=':',
                    label=f'Median: {df["centroid_z"].median():.1f}')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig11.savefig(f'{output_dir}/centroid_z_distribution.png', dpi=300, bbox_inches='tight')
        plt.close(fig11)

    # 12. 点数量分布直方图（如果存在）
    if 'point_count' in df.columns:
        fig12 = plt.figure(figsize=(10, 8))
        plt.hist(df['point_count'], bins=30, edgecolor='black', alpha=0.7, color='gold')
        plt.xlabel('Point Count', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title('Point Count Distribution', fontsize=16, fontweight='bold')
        plt.axvline(df['point_count'].mean(), color='red', linestyle='--',
                    label=f'Mean: {df["point_count"].mean():.1f}')
        plt.axvline(df['point_count'].median(), color='blue', linestyle=':',
                    label=f'Median: {df["point_count"].median():.1f}')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.tight_layout()
        fig12.savefig(f'{output_dir}/point_count_distribution.png', dpi=300, bbox_inches='tight')
        plt.close(fig12)

    # 13. 点数量累积分布函数
    if 'point_count' in df.columns:
        fig13 = plt.figure(figsize=(10, 8))
        sorted_counts = np.sort(df['point_count'])
        y_vals = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
        plt.plot(sorted_counts, y_vals, 'b-', linewidth=2)
        plt.xlabel('Point Count', fontsize=14)
        plt.ylabel('Cumulative Probability', fontsize=14)
        plt.title('Point Count CDF', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xscale('log')

        # 标记关键分位数
        for quantile in [0.25, 0.5, 0.75, 0.9]:
            idx = int(quantile * len(sorted_counts))
            if idx < len(sorted_counts):
                plt.axvline(sorted_counts[idx], color='red', linestyle='--', alpha=0.5)
                plt.text(sorted_counts[idx] * 1.1, quantile - 0.05,
                         f'{quantile * 100:.0f}%: {sorted_counts[idx]:.0f}',
                         fontsize=9, color='red')

        plt.tight_layout()
        fig13.savefig(f'{output_dir}/point_count_cdf.png', dpi=300, bbox_inches='tight')
        plt.close(fig13)

    # 保存核心统计摘要
    summary_file = os.path.join(output_dir, "core_statistics_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("孔隙聚类核心统计摘要\n")
        f.write("=" * 50 + "\n")
        f.write(f"总孔隙数: {len(df):,}\n\n")

        if 'equivalent_radius' in df.columns:
            f.write("等效半径统计:\n")
            f.write(f"  平均值: {df['equivalent_radius'].mean():.4f}\n")
            f.write(f"  中位数: {df['equivalent_radius'].median():.4f}\n")
            f.write(f"  最小值: {df['equivalent_radius'].min():.4f}\n")
            f.write(f"  最大值: {df['equivalent_radius'].max():.4f}\n")
            f.write(f"  标准差: {df['equivalent_radius'].std():.4f}\n\n")

        if 'point_count' in df.columns:
            f.write("孔隙大小统计:\n")
            f.write(f"  平均点数: {df['point_count'].mean():.1f}\n")
            f.write(f"  中位数点数: {df['point_count'].median():.1f}\n")
            f.write(f"  最小点数: {df['point_count'].min()}\n")
            f.write(f"  最大点数: {df['point_count'].max()}\n")
            f.write(f"  总点数: {df['point_count'].sum():,}\n\n")

        if all(col in df.columns for col in ['centroid_x', 'centroid_y', 'centroid_z']):
            f.write("质心坐标范围:\n")
            f.write(f"  X: [{df['centroid_x'].min():.1f}, {df['centroid_x'].max():.1f}]\n")
            f.write(f"  Y: [{df['centroid_y'].min():.1f}, {df['centroid_y'].max():.1f}]\n")
            f.write(f"  Z: [{df['centroid_z'].min():.1f}, {df['centroid_z'].max():.1f}]\n")

            f.write("\n质心坐标统计:\n")
            f.write(f"  X平均值: {df['centroid_x'].mean():.1f}, 标准差: {df['centroid_x'].std():.1f}\n")
            f.write(f"  Y平均值: {df['centroid_y'].mean():.1f}, 标准差: {df['centroid_y'].std():.1f}\n")
            f.write(f"  Z平均值: {df['centroid_z'].mean():.1f}, 标准差: {df['centroid_z'].std():.1f}\n")

    return df


def plot_plane_features_statistics(csv_file_path, output_dir="plane_features_statistics"):
    """
    读取平面特征信息CSV文件并绘制独立的统计图

    参数:
        csv_file_path: CSV文件路径
        output_dir: 输出图像目录
    """

    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取数据
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return None

    # 2. 计算衍生特征
    if 'Length' in df.columns and 'Width' in df.columns:
        df['Area'] = df['Length'] * df['Width']
        df['Aspect_Ratio'] = df['Length'] / df['Width']

    # 3. 创建独立的图形

    # 图1: 平面面积分布直方图
    if 'Area' in df.columns:
        fig1 = plt.figure(figsize=(10, 8))
        plt.hist(df['Area'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        plt.xlabel('Plane Area', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title('Plane Area Distribution', fontsize=16, fontweight='bold')
        plt.axvline(df['Area'].mean(), color='red', linestyle='--',
                    label=f'Mean: {df["Area"].mean():.2f}')
        plt.axvline(df['Area'].median(), color='blue', linestyle=':',
                    label=f'Median: {df["Area"].median():.2f}')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig1.savefig(f'{output_dir}/plane_area_histogram.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)

    # 图2: 平面面积箱线图
    if 'Area' in df.columns:
        fig2 = plt.figure(figsize=(8, 10))
        plt.boxplot(df['Area'], vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightcoral'))
        plt.ylabel('Plane Area', fontsize=14)
        plt.title('Plane Area Box Plot', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig2.savefig(f'{output_dir}/plane_area_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)

    # 图3: 长度分布直方图
    if 'Length' in df.columns:
        fig3 = plt.figure(figsize=(10, 8))
        plt.hist(df['Length'], bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
        plt.xlabel('Length', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title('Plane Length Distribution', fontsize=16, fontweight='bold')
        plt.axvline(df['Length'].mean(), color='red', linestyle='--',
                    label=f'Mean: {df["Length"].mean():.2f}')
        plt.axvline(df['Length'].median(), color='blue', linestyle=':',
                    label=f'Median: {df["Length"].median():.2f}')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig3.savefig(f'{output_dir}/plane_length_histogram.png', dpi=300, bbox_inches='tight')
        plt.close(fig3)

    # 图4: 宽度分布直方图
    if 'Width' in df.columns:
        fig4 = plt.figure(figsize=(10, 8))
        plt.hist(df['Width'], bins=30, edgecolor='black', alpha=0.7, color='gold')
        plt.xlabel('Width', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title('Plane Width Distribution', fontsize=16, fontweight='bold')
        plt.axvline(df['Width'].mean(), color='red', linestyle='--',
                    label=f'Mean: {df["Width"].mean():.2f}')
        plt.axvline(df['Width'].median(), color='blue', linestyle=':',
                    label=f'Median: {df["Width"].median():.2f}')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig4.savefig(f'{output_dir}/plane_width_histogram.png', dpi=300, bbox_inches='tight')
        plt.close(fig4)

    # 图5: 长宽比分布直方图
    if 'Aspect_Ratio' in df.columns:
        fig5 = plt.figure(figsize=(10, 8))
        plt.hist(df['Aspect_Ratio'], bins=30, edgecolor='black', alpha=0.7, color='violet')
        plt.xlabel('Aspect Ratio (Length/Width)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title('Plane Aspect Ratio Distribution', fontsize=16, fontweight='bold')
        plt.axvline(df['Aspect_Ratio'].mean(), color='red', linestyle='--',
                    label=f'Mean: {df["Aspect_Ratio"].mean():.2f}')
        plt.axvline(df['Aspect_Ratio'].median(), color='blue', linestyle=':',
                    label=f'Median: {df["Aspect_Ratio"].median():.2f}')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig5.savefig(f'{output_dir}/plane_aspect_ratio_histogram.png', dpi=300, bbox_inches='tight')
        plt.close(fig5)

    # 图6: 长度与宽度关系散点图
    if 'Length' in df.columns and 'Width' in df.columns:
        fig6 = plt.figure(figsize=(10, 8))
        plt.scatter(df['Length'], df['Width'], alpha=0.6, s=20, color='darkorange')
        plt.xlabel('Length', fontsize=14)
        plt.ylabel('Width', fontsize=14)
        plt.title('Length vs Width Scatter Plot', fontsize=16, fontweight='bold')

        # 添加对角线
        max_val = max(df['Length'].max(), df['Width'].max())
        plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Length = Width')

        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig6.savefig(f'{output_dir}/length_vs_width_scatter.png', dpi=300, bbox_inches='tight')
        plt.close(fig6)

    # 图7: 面积累积分布函数
    if 'Area' in df.columns:
        fig7 = plt.figure(figsize=(10, 8))
        sorted_area = np.sort(df['Area'])
        y_vals = np.arange(1, len(sorted_area) + 1) / len(sorted_area)
        plt.plot(sorted_area, y_vals, 'b-', linewidth=2)
        plt.xlabel('Plane Area', fontsize=14)
        plt.ylabel('Cumulative Probability', fontsize=14)
        plt.title('Plane Area CDF', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # 标记关键分位数
        for quantile in [0.25, 0.5, 0.75, 0.9]:
            idx = int(quantile * len(sorted_area))
            if idx < len(sorted_area):
                plt.axvline(sorted_area[idx], color='red', linestyle='--', alpha=0.5)
                plt.text(sorted_area[idx] * 1.1, quantile - 0.05,
                         f'{quantile * 100:.0f}%: {sorted_area[idx]:.2f}',
                         fontsize=9, color='red')

        plt.tight_layout()
        fig7.savefig(f'{output_dir}/plane_area_cdf.png', dpi=300, bbox_inches='tight')
        plt.close(fig7)

    # 图8: 三维中心点分布
    if all(col in df.columns for col in ['Center_X', 'Center_Y', 'Center_Z']):
        fig8 = plt.figure(figsize=(12, 10))
        ax8 = fig8.add_subplot(111, projection='3d')

        if 'Area' in df.columns:
            # 根据面积着色
            scatter8 = ax8.scatter(df['Center_X'], df['Center_Y'], df['Center_Z'],
                                   c=df['Area'], alpha=0.6, s=15, cmap='viridis')
            plt.colorbar(scatter8, ax=ax8, shrink=0.8, label='Plane Area')
        else:
            ax8.scatter(df['Center_X'], df['Center_Y'], df['Center_Z'],
                        alpha=0.6, s=15, color='blue')

        ax8.set_xlabel('X Coordinate', fontsize=14, labelpad=10)
        ax8.set_ylabel('Y Coordinate', fontsize=14, labelpad=10)
        ax8.set_zlabel('Z Coordinate', fontsize=14, labelpad=10)
        ax8.set_title('3D Plane Center Distribution', fontsize=16, fontweight='bold', pad=20)
        ax8.view_init(elev=20, azim=45)
        plt.tight_layout()
        fig8.savefig(f'{output_dir}/3d_plane_center_distribution.png', dpi=300, bbox_inches='tight')
        plt.close(fig8)

    # 图9: X-Y平面投影
    if all(col in df.columns for col in ['Center_X', 'Center_Y']):
        fig9 = plt.figure(figsize=(10, 8))
        ax9 = fig9.add_subplot(111)

        if 'Area' in df.columns:
            scatter9 = ax9.scatter(df['Center_X'], df['Center_Y'],
                                   c=df['Area'], alpha=0.6, s=15, cmap='viridis')
            plt.colorbar(scatter9, ax=ax9, shrink=0.8, label='Plane Area')
        else:
            ax9.scatter(df['Center_X'], df['Center_Y'], alpha=0.6, s=15, color='blue')

        ax9.set_xlabel('X Coordinate', fontsize=14)
        ax9.set_ylabel('Y Coordinate', fontsize=14)
        ax9.set_title('Plane Center X-Y Projection', fontsize=16, fontweight='bold')
        ax9.grid(True, alpha=0.3)
        plt.tight_layout()
        fig9.savefig(f'{output_dir}/plane_center_xy_projection.png', dpi=300, bbox_inches='tight')
        plt.close(fig9)

    # 图10: X-Z平面投影
    if all(col in df.columns for col in ['Center_X', 'Center_Z']):
        fig10 = plt.figure(figsize=(10, 8))
        ax10 = fig10.add_subplot(111)

        if 'Area' in df.columns:
            scatter10 = ax10.scatter(df['Center_X'], df['Center_Z'],
                                     c=df['Area'], alpha=0.6, s=15, cmap='viridis')
            plt.colorbar(scatter10, ax=ax10, shrink=0.8, label='Plane Area')
        else:
            ax10.scatter(df['Center_X'], df['Center_Z'], alpha=0.6, s=15, color='blue')

        ax10.set_xlabel('X Coordinate', fontsize=14)
        ax10.set_ylabel('Z Coordinate', fontsize=14)
        ax10.set_title('Plane Center X-Z Projection', fontsize=16, fontweight='bold')
        ax10.grid(True, alpha=0.3)
        plt.tight_layout()
        fig10.savefig(f'{output_dir}/plane_center_xz_projection.png', dpi=300, bbox_inches='tight')
        plt.close(fig10)

    # 图11: Y-Z平面投影
    if all(col in df.columns for col in ['Center_Y', 'Center_Z']):
        fig11 = plt.figure(figsize=(10, 8))
        ax11 = fig11.add_subplot(111)

        if 'Area' in df.columns:
            scatter11 = ax11.scatter(df['Center_Y'], df['Center_Z'],
                                     c=df['Area'], alpha=0.6, s=15, cmap='viridis')
            plt.colorbar(scatter11, ax=ax11, shrink=0.8, label='Plane Area')
        else:
            ax11.scatter(df['Center_Y'], df['Center_Z'], alpha=0.6, s=15, color='blue')

        ax11.set_xlabel('Y Coordinate', fontsize=14)
        ax11.set_ylabel('Z Coordinate', fontsize=14)
        ax11.set_title('Plane Center Y-Z Projection', fontsize=16, fontweight='bold')
        ax11.grid(True, alpha=0.3)
        plt.tight_layout()
        fig11.savefig(f'{output_dir}/plane_center_yz_projection.png', dpi=300, bbox_inches='tight')
        plt.close(fig11)

    # 图12: 中心点X坐标分布直方图
    if 'Center_X' in df.columns:
        fig12 = plt.figure(figsize=(10, 8))
        plt.hist(df['Center_X'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        plt.xlabel('X Coordinate', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title('Plane Center X Distribution', fontsize=16, fontweight='bold')
        plt.axvline(df['Center_X'].mean(), color='red', linestyle='--',
                    label=f'Mean: {df["Center_X"].mean():.2f}')
        plt.axvline(df['Center_X'].median(), color='blue', linestyle=':',
                    label=f'Median: {df["Center_X"].median():.2f}')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig12.savefig(f'{output_dir}/plane_center_x_distribution.png', dpi=300, bbox_inches='tight')
        plt.close(fig12)

    # 图13: 中心点Y坐标分布直方图
    if 'Center_Y' in df.columns:
        fig13 = plt.figure(figsize=(10, 8))
        plt.hist(df['Center_Y'], bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
        plt.xlabel('Y Coordinate', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title('Plane Center Y Distribution', fontsize=16, fontweight='bold')
        plt.axvline(df['Center_Y'].mean(), color='red', linestyle='--',
                    label=f'Mean: {df["Center_Y"].mean():.2f}')
        plt.axvline(df['Center_Y'].median(), color='blue', linestyle=':',
                    label=f'Median: {df["Center_Y"].median():.2f}')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig13.savefig(f'{output_dir}/plane_center_y_distribution.png', dpi=300, bbox_inches='tight')
        plt.close(fig13)

    # 图14: 中心点Z坐标分布直方图
    if 'Center_Z' in df.columns:
        fig14 = plt.figure(figsize=(10, 8))
        plt.hist(df['Center_Z'], bins=30, edgecolor='black', alpha=0.7, color='violet')
        plt.xlabel('Z Coordinate', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title('Plane Center Z Distribution', fontsize=16, fontweight='bold')
        plt.axvline(df['Center_Z'].mean(), color='red', linestyle='--',
                    label=f'Mean: {df["Center_Z"].mean():.2f}')
        plt.axvline(df['Center_Z'].median(), color='blue', linestyle=':',
                    label=f'Median: {df["Center_Z"].median():.2f}')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig14.savefig(f'{output_dir}/plane_center_z_distribution.png', dpi=300, bbox_inches='tight')
        plt.close(fig14)

    # 图15: 法向量分布（单位球面投影）
    if all(col in df.columns for col in ['Normal_X', 'Normal_Y', 'Normal_Z']):
        # 计算法向量的方位角和俯仰角
        df['Normal_Theta'] = np.arctan2(df['Normal_Y'], df['Normal_X'])  # 方位角
        df['Normal_Phi'] = np.arccos(np.abs(df['Normal_Z']))  # 俯仰角

        fig15 = plt.figure(figsize=(12, 10))
        ax15 = fig15.add_subplot(111, projection='polar')

        # 转换为极坐标
        scatter15 = ax15.scatter(df['Normal_Theta'], df['Normal_Phi'],
                                 alpha=0.6, s=20, c=df['Area'] if 'Area' in df.columns else 'blue',
                                 cmap='viridis' if 'Area' in df.columns else None)

        ax15.set_title('Normal Vector Distribution on Unit Sphere', fontsize=16, fontweight='bold')
        ax15.set_theta_zero_location('E')
        ax15.set_theta_direction(-1)
        ax15.set_rlabel_position(22.5)
        ax15.set_rticks([0, np.pi / 6, np.pi / 3, np.pi / 2])
        ax15.set_yticklabels(['0°', '30°', '60°', '90°'])
        ax15.set_ylim(0, np.pi / 2)
        if 'Area' in df.columns:
            plt.colorbar(scatter15, ax=ax15, shrink=0.8, label='Plane Area')

        plt.tight_layout()
        fig15.savefig(f'{output_dir}/normal_vector_polar.png', dpi=300, bbox_inches='tight')
        plt.close(fig15)

    # 图16: 法向量X分量分布
    if 'Normal_X' in df.columns:
        fig16 = plt.figure(figsize=(10, 8))
        plt.hist(df['Normal_X'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        plt.xlabel('Normal X Component', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title('Normal Vector X Component Distribution', fontsize=16, fontweight='bold')
        plt.axvline(df['Normal_X'].mean(), color='red', linestyle='--',
                    label=f'Mean: {df["Normal_X"].mean():.3f}')
        plt.axvline(df['Normal_X'].median(), color='blue', linestyle=':',
                    label=f'Median: {df["Normal_X"].median():.3f}')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig16.savefig(f'{output_dir}/normal_x_component_histogram.png', dpi=300, bbox_inches='tight')
        plt.close(fig16)

    # 图17: 法向量Y分量分布
    if 'Normal_Y' in df.columns:
        fig17 = plt.figure(figsize=(10, 8))
        plt.hist(df['Normal_Y'], bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
        plt.xlabel('Normal Y Component', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title('Normal Vector Y Component Distribution', fontsize=16, fontweight='bold')
        plt.axvline(df['Normal_Y'].mean(), color='red', linestyle='--',
                    label=f'Mean: {df["Normal_Y"].mean():.3f}')
        plt.axvline(df['Normal_Y'].median(), color='blue', linestyle=':',
                    label=f'Median: {df["Normal_Y"].median():.3f}')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig17.savefig(f'{output_dir}/normal_y_component_histogram.png', dpi=300, bbox_inches='tight')
        plt.close(fig17)

    # 图18: 法向量Z分量分布
    if 'Normal_Z' in df.columns:
        fig18 = plt.figure(figsize=(10, 8))
        plt.hist(df['Normal_Z'], bins=30, edgecolor='black', alpha=0.7, color='violet')
        plt.xlabel('Normal Z Component', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title('Normal Vector Z Component Distribution', fontsize=16, fontweight='bold')
        plt.axvline(df['Normal_Z'].mean(), color='red', linestyle='--',
                    label=f'Mean: {df["Normal_Z"].mean():.3f}')
        plt.axvline(df['Normal_Z'].median(), color='blue', linestyle=':',
                    label=f'Median: {df["Normal_Z"].median():.3f}')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig18.savefig(f'{output_dir}/normal_z_component_histogram.png', dpi=300, bbox_inches='tight')
        plt.close(fig18)

    # 图19: 特征相关性热图
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        fig20 = plt.figure(figsize=(12, 10))
        correlation_matrix = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f',
                    cmap='coolwarm', center=0, square=True, linewidths=.5,
                    cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        fig20.savefig(f'{output_dir}/feature_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close(fig20)

    # 保存核心统计摘要
    summary_file = os.path.join(output_dir, "plane_statistics_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("平面特征核心统计摘要\n")
        f.write("=" * 60 + "\n")
        f.write(f"总平面数: {len(df):,}\n\n")

        # 平面尺寸统计
        if 'Length' in df.columns and 'Width' in df.columns:
            f.write("平面尺寸统计:\n")
            f.write(f"  平均长度: {df['Length'].mean():.2f}\n")
            f.write(f"  长度中位数: {df['Length'].median():.2f}\n")
            f.write(f"  长度范围: [{df['Length'].min():.2f}, {df['Length'].max():.2f}]\n")
            f.write(f"  平均宽度: {df['Width'].mean():.2f}\n")
            f.write(f"  宽度中位数: {df['Width'].median():.2f}\n")
            f.write(f"  宽度范围: [{df['Width'].min():.2f}, {df['Width'].max():.2f}]\n\n")

        # 面积统计
        if 'Area' in df.columns:
            f.write("平面面积统计:\n")
            f.write(f"  平均面积: {df['Area'].mean():.2f}\n")
            f.write(f"  面积中位数: {df['Area'].median():.2f}\n")
            f.write(f"  面积范围: [{df['Area'].min():.2f}, {df['Area'].max():.2f}]\n")
            f.write(f"  总面积: {df['Area'].sum():.2f}\n")
            f.write(f"  面积标准差: {df['Area'].std():.2f}\n\n")

        # 长宽比统计
        if 'Aspect_Ratio' in df.columns:
            f.write("长宽比统计:\n")
            f.write(f"  平均长宽比: {df['Aspect_Ratio'].mean():.2f}\n")
            f.write(f"  长宽比中位数: {df['Aspect_Ratio'].median():.2f}\n")
            f.write(f"  长宽比范围: [{df['Aspect_Ratio'].min():.2f}, {df['Aspect_Ratio'].max():.2f}]\n\n")

        # 中心点坐标统计
        if all(col in df.columns for col in ['Center_X', 'Center_Y', 'Center_Z']):
            f.write("中心点坐标统计:\n")
            f.write(f"  X坐标范围: [{df['Center_X'].min():.2f}, {df['Center_X'].max():.2f}]\n")
            f.write(f"  Y坐标范围: [{df['Center_Y'].min():.2f}, {df['Center_Y'].max():.2f}]\n")
            f.write(f"  Z坐标范围: [{df['Center_Z'].min():.2f}, {df['Center_Z'].max():.2f}]\n")
            f.write(f"  X平均值: {df['Center_X'].mean():.2f}, 标准差: {df['Center_X'].std():.2f}\n")
            f.write(f"  Y平均值: {df['Center_Y'].mean():.2f}, 标准差: {df['Center_Y'].std():.2f}\n")
            f.write(f"  Z平均值: {df['Center_Z'].mean():.2f}, 标准差: {df['Center_Z'].std():.2f}\n\n")

        # 法向量统计
        if all(col in df.columns for col in ['Normal_X', 'Normal_Y', 'Normal_Z']):
            f.write("法向量统计:\n")
            f.write(f"  X分量范围: [{df['Normal_X'].min():.3f}, {df['Normal_X'].max():.3f}]\n")
            f.write(f"  Y分量范围: [{df['Normal_Y'].min():.3f}, {df['Normal_Y'].max():.3f}]\n")
            f.write(f"  Z分量范围: [{df['Normal_Z'].min():.3f}, {df['Normal_Z'].max():.3f}]\n")

            # 计算法向量主要方向
            if 'Normal_Z' in df.columns:
                vertical_planes = len(df[abs(df['Normal_Z']) > 0.9])
                horizontal_planes = len(df[abs(df['Normal_Z']) < 0.3])
                f.write(
                    f"  近似垂直平面 (|Normal_Z| > 0.9): {vertical_planes}个 ({vertical_planes / len(df) * 100:.1f}%)\n")
                f.write(
                    f"  近似水平平面 (|Normal_Z| < 0.3): {horizontal_planes}个 ({horizontal_planes / len(df) * 100:.1f}%)\n")

    # print(f"所有统计图已保存到目录: {output_dir}")
    # print(f"统计摘要已保存到: {summary_file}")

    return df



# 使用示例
if __name__ == "__main__":
    # # 调用函数绘制图像
    # csv_file = "C:\\Users\\dell\\PycharmProjects\\Mask2AM\\porosity_cluster_info.csv"
    # df = plot_porosity_statistics(csv_file, output_dir="pore_statistics")
    #
    # # 如果需要进一步分析，可以使用返回的DataFrame
    # if df is not None:
    #     print("\n数据概览:")
    #     print(df[['equivalent_radius', 'point_count']].describe())


    plot_plane_features_statistics("C:\\Users\\dell\\PycharmProjects\\Mask2AM\\patches_info.csv")