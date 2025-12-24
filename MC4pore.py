from pylab import mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
from skimage import measure, io
from tqdm import tqdm
import plotly.graph_objects as go
from scipy import ndimage
import json
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class PoreReconstruction3D:
    """基于分割二值图像的三维孔隙重建"""

    def __init__(self, voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """
        参数:
            voxel_size: 体素大小 (z, y, x) 单位，用于设置真实尺度
        """
        self.volume_data = None
        self.binary_volume = None
        self.mesh_vertices = None
        self.mesh_faces = None
        self.voxel_size = voxel_size
        self.labeled_volume = None  # 添加标记体积缓存

    def load_binary_images(self, folder_path: str, file_pattern: str = "*.png") -> np.ndarray:
        """加载已经分割好的二值图像序列"""
        image_files = sorted(list(Path(folder_path).glob(file_pattern)))
        if not image_files:
            raise ValueError(f"\n在路径 {folder_path} 中没有找到图像文件")

        slices = []
        for img_file in tqdm(image_files, desc="加载二值图像"):
            img = io.imread(str(img_file))
            if img.max() > 1:
                binary_img = (img > 0).astype(np.uint8) * 255
            else:
                binary_img = img.astype(np.uint8) * 255
            slices.append(binary_img)

        self.volume_data = np.stack(slices, axis=0)
        self.binary_volume = self.volume_data
        return self.volume_data

    def marching_cubes(self, level: float = 128) -> Tuple[np.ndarray, np.ndarray]:
        """Marching Cubes三维重建"""
        if self.binary_volume is None:
            raise ValueError("\n请先加载图像数据")

        volume = self.binary_volume.astype(np.float32) / 255.0
        verts, faces, normals, values = measure.marching_cubes(
            volume=volume,
            level=level / 255.0,
            spacing=self.voxel_size,
            step_size=5, #设置步长
            allow_degenerate=False,
            method='lewiner'
        )

        self.mesh_vertices = verts
        self.mesh_faces = faces

        print(f"\nMarching Cubes重建:")
        print(f"  - 顶点数: {len(verts)}")
        print(f"  - 面数: {len(faces)}")

        return verts, faces

    def calculate_porosity(self) -> Dict[str, float]:
        """计算三维孔隙率"""
        if self.binary_volume is None:
            raise ValueError("\n请先加载图像数据")

        pore_voxels = np.sum(self.binary_volume > 0)
        total_voxels = self.binary_volume.size
        porosity = (pore_voxels / total_voxels) * 100

        results = {
            'porosity_3d_percent': float(porosity),
            'porosity_fraction': float(pore_voxels / total_voxels),
            'pore_volume_voxels': int(pore_voxels),
            'total_volume_voxels': int(total_voxels),
        }

        print(f"\n三维孔隙分析:")
        print(f"  - 孔隙率: {porosity:.2f}%")
        print(f"  - 孔隙体积: {pore_voxels} 体素")

        return results

    def analyze_pore_size_distribution(self) -> Dict:
        """分析孔隙尺寸分布"""
        if self.binary_volume is None:
            raise ValueError("\n请先加载图像数据")

        if self.labeled_volume is None:
            self.labeled_volume, num_pores = ndimage.label(self.binary_volume > 0)
        else:
            num_pores = np.max(self.labeled_volume)

        if num_pores == 0:
            print("\n未检测到孔隙")
            return {"total_pores": 0}

        pore_indices = np.arange(1, num_pores + 1)
        pore_volumes = ndimage.sum(self.binary_volume > 0, self.labeled_volume, pore_indices)
        pore_diameters = 2 * ((3 * pore_volumes * np.prod(self.voxel_size)) / (4 * np.pi)) ** (1 / 3)

        if pore_diameters.min() <= 0:
            pore_diameters = np.clip(pore_diameters, 1e-6, None)
        if pore_volumes.min() <= 0:
            pore_volumes = np.clip(pore_volumes, 1e-6, None)

        volume_percentiles = np.percentile(pore_volumes, [10, 25, 50, 75, 90, 95, 99])
        diameter_percentiles = np.percentile(pore_diameters, [10, 25, 50, 75, 90, 95, 99])
        volume_weighted_diameter = np.sum(pore_volumes * pore_diameters) / np.sum(pore_volumes) if np.sum(
            pore_volumes) > 0 else 0

        diameter_bins = np.logspace(np.log10(pore_diameters.min()), np.log10(pore_diameters.max()), 20)
        volume_bins = np.logspace(np.log10(pore_volumes.min()), np.log10(pore_volumes.max()), 20)
        diameter_hist, diameter_bin_edges = np.histogram(pore_diameters, bins=diameter_bins)
        volume_hist, volume_bin_edges = np.histogram(pore_volumes, bins=volume_bins)

        sorted_diameters = np.sort(pore_diameters)
        cumulative_dist = np.arange(1, len(sorted_diameters) + 1) / len(sorted_diameters)

        stats = {
            'total_pores': int(num_pores),
            'total_pore_volume': float(np.sum(pore_volumes)),
            'avg_pore_volume': float(np.mean(pore_volumes)),
            'std_pore_volume': float(np.std(pore_volumes)),
            'avg_pore_diameter': float(np.mean(pore_diameters)),
            'std_pore_diameter': float(np.std(pore_diameters)),
            'volume_weighted_diameter': float(volume_weighted_diameter),
            'max_pore_diameter': float(np.max(pore_diameters)),
            'min_pore_diameter': float(np.min(pore_diameters)),
            'pore_volume_distribution': pore_volumes.tolist(),
            'pore_diameter_distribution': pore_diameters.tolist(),
            'volume_percentiles': volume_percentiles.tolist(),
            'diameter_percentiles': diameter_percentiles.tolist(),
            'diameter_histogram': {
                'counts': diameter_hist.tolist(),
                'bin_edges': diameter_bin_edges.tolist()
            },
            'volume_histogram': {
                'counts': volume_hist.tolist(),
                'bin_edges': volume_bin_edges.tolist()
            },
            'cumulative_distribution': {
                'diameters': sorted_diameters.tolist(),
                'cumulative': cumulative_dist.tolist()
            }
        }

        print(f"\n孔隙尺寸分布:")
        print(f"  - 孔隙总数: {num_pores:,}")
        print(f"  - 平均孔隙直径: {stats['avg_pore_diameter']:.2f} 单位")
        print(f"  - 体积加权平均直径: {stats['volume_weighted_diameter']:.2f} 单位")
        print(f"  - 最大孔隙直径: {stats['max_pore_diameter']:.2f} 单位")
        print(f"  - 最小孔隙直径: {stats['min_pore_diameter']:.2f} 单位")
        print(f"  - 孔隙直径中位数: {stats['diameter_percentiles'][2]:.2f} 单位")
        print(f"  - 孔隙直径90%分位数: {stats['diameter_percentiles'][4]:.2f} 单位")
        print(f"  - 总孔隙体积: {stats['total_pore_volume']:,} 体素")

        return stats

    def export_mesh(self, filename: str, format: str = 'ply'):
        """导出网格文件"""
        if self.mesh_vertices is None or self.mesh_faces is None:
            raise ValueError("\n请先进行三维重建")

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        if format.lower() == 'ply':
            self._export_ply(filename)
        elif format.lower() == 'stl':
            self._export_stl(filename)
        elif format.lower() == 'obj':
            self._export_obj(filename)
        else:
            raise ValueError(f"\n不支持的格式: {format}")

    def _export_ply(self, filename: str):
        """导出为PLY格式"""
        print(f"导出PLY网格文件: {filename}")

        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(self.mesh_vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(self.mesh_faces)}\n")
            f.write("property list uchar int vertex_index\n")
            f.write("end_header\n")

            batch_size = 100000
            for i in tqdm(range(0, len(self.mesh_vertices), batch_size),
                          desc="导出顶点", unit="批"):
                batch = self.mesh_vertices[i:i + batch_size]
                lines = [f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in batch]
                f.write("\n".join(lines) + "\n")

            for i in tqdm(range(0, len(self.mesh_faces), batch_size),
                          desc="导出面", unit="批"):
                batch = self.mesh_faces[i:i + batch_size]
                lines = [f"3 {f[0]} {f[1]} {f[2]}" for f in batch]
                f.write("\n".join(lines) + "\n")

        print(f"\n网格已导出为: {filename}")

    def _export_stl(self, filename: str):
        """导出为STL格式 - 仅适用于小数据集"""
        try:
            from stl import mesh
        except ImportError:
            print("\n需要安装numpy-stl库: pip install numpy-stl")
            return

        print(f"导出STL网格文件: {filename}")

        if len(self.mesh_faces) > 1000000:
            print(f"警告: 网格面数过多 ({len(self.mesh_faces)}), STL导出可能非常慢且占用大量内存")
            print("建议使用PLY格式导出")

        stl_mesh = mesh.Mesh(np.zeros(len(self.mesh_faces), dtype=mesh.Mesh.dtype))
        for i in tqdm(range(len(self.mesh_faces)), desc="生成STL网格", unit="面"):
            face = self.mesh_faces[i]
            for j in range(3):
                stl_mesh.vectors[i][j] = self.mesh_vertices[face[j]]

        stl_mesh.save(filename)
        print(f"网格已导出为: {filename}")

    def _export_obj(self, filename: str):
        """导出为OBJ格式 - 分批写入"""
        print(f"\n导出OBJ网格文件: {filename}")

        with open(filename, 'w') as f:
            f.write("# 孔隙结构三维模型\n")

            batch_size = 100000
            for i in tqdm(range(0, len(self.mesh_vertices), batch_size),
                          desc="导出顶点", unit="批"):
                batch = self.mesh_vertices[i:i + batch_size]
                lines = [f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in batch]
                f.write("\n".join(lines) + "\n")

            for i in tqdm(range(0, len(self.mesh_faces), batch_size),
                          desc="导出面", unit="批"):
                batch = self.mesh_faces[i:i + batch_size]
                lines = [f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}" for f in batch]
                f.write("\n".join(lines) + "\n")

        print(f"\n网格已导出为: {filename}")

    def visualize_3d(self, save_path: str = None, show: bool = False):
        """Marching Cubes结果的3D可视化"""
        if self.mesh_vertices is None or self.mesh_faces is None:
            raise ValueError("\n请先进行三维重建")

        fig = go.Figure()

        # 1. 先添加实体面
        fig.add_trace(go.Mesh3d(
            x=self.mesh_vertices[:, 0],
            y=self.mesh_vertices[:, 1],
            z=self.mesh_vertices[:, 2],
            i=self.mesh_faces[:, 0],
            j=self.mesh_faces[:, 1],
            k=self.mesh_faces[:, 2],
            color='#87CEEB',
            opacity=1,
            flatshading=True,
            lighting=dict(
                ambient=0.9,
                diffuse=0.6,
                specular=0.05,
                roughness=0.9,
                fresnel=0.1
            ),
            lightposition=dict(x=100, y=100, z=1000),
            hoverinfo='none',
            showscale=False,
            name='孔隙实体'
        ))

        # 2. 添加线框
        edges = set()
        for face in self.mesh_faces:
            edges.add(tuple(sorted([face[0], face[1]])))
            edges.add(tuple(sorted([face[1], face[2]])))
            edges.add(tuple(sorted([face[2], face[0]])))

        # 创建线的坐标
        line_x = []
        line_y = []
        line_z = []

        for edge in edges:
            v1, v2 = edge
            # 第一条边
            line_x.append(self.mesh_vertices[v1, 0])
            line_y.append(self.mesh_vertices[v1, 1])
            line_z.append(self.mesh_vertices[v1, 2])

            # 第二条边
            line_x.append(self.mesh_vertices[v2, 0])
            line_y.append(self.mesh_vertices[v2, 1])
            line_z.append(self.mesh_vertices[v2, 2])

            # 添加None以断开线段
            line_x.append(None)
            line_y.append(None)
            line_z.append(None)

        # 添加线框轨迹
        fig.add_trace(go.Scatter3d(
            x=line_x,
            y=line_y,
            z=line_z,
            mode='lines',
            line=dict(
                color='#333333',
                width=2.0
            ),
            opacity=1,
            hoverinfo='none',
            name='孔隙边界'
        ))

        # 设置布局
        fig.update_layout(
            title='三维孔隙结构重建',
            scene=dict(
                xaxis_title='X (μm)',
                yaxis_title='Y (μm)',
                zaxis_title='Z (μm)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.0),  # 调整视角
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                ),
                xaxis=dict(
                    showbackground=True,
                    backgroundcolor="#f8f9fa",
                    gridcolor="#e9ecef",
                    showticklabels=True
                ),
                yaxis=dict(
                    showbackground=True,
                    backgroundcolor="#f8f9fa",
                    gridcolor="#e9ecef",
                    showticklabels=True
                ),
                zaxis=dict(
                    showbackground=True,
                    backgroundcolor="#f8f9fa",
                    gridcolor="#e9ecef",
                    showticklabels=True
                ),
            ),
            width=1200,  # 调整尺寸
            height=1000,
            paper_bgcolor='white',
            margin=dict(l=0, r=0, t=60, b=0),
            showlegend=False  # 显示图例
        )

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_html(
                save_path,
                include_plotlyjs='cdn',
                full_html=False,
                config=dict(responsive=True, displayModeBar=True)  # 显示工具栏
            )
            print(f"\n三维可视化已保存至: {save_path}")

        if show:
            fig.show()

    def plot_pore_size_distribution(self, save_path: str = None):
        """绘制孔隙尺寸分布图"""
        if self.binary_volume is None:
            raise ValueError("\n请先加载图像数据")

        stats = self.analyze_pore_size_distribution()

        if stats['total_pores'] == 0:
            print("无孔隙数据可绘制")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        diameters = stats['pore_diameter_distribution']
        ax1 = axes[0, 0]
        ax1.hist(diameters, bins=50, alpha=0.7, color='steelblue', edgecolor='navy', linewidth=0.5)
        ax1.set_xlabel('孔隙直径 (单位)', fontsize=12)
        ax1.set_ylabel('孔隙数量', fontsize=12)
        ax1.set_title('孔隙直径分布直方图', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xscale('log')
        ax1.set_yscale('log')

        volumes = stats['pore_volume_distribution']
        ax2 = axes[0, 1]
        ax2.hist(volumes, bins=50, alpha=0.7, color='forestgreen', edgecolor='darkgreen', linewidth=0.5)
        ax2.set_xlabel('孔隙体积 (体素)', fontsize=12)
        ax2.set_ylabel('孔隙数量', fontsize=12)
        ax2.set_title('孔隙体积分布直方图', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xscale('log')
        ax2.set_yscale('log')

        cumulative_data = stats['cumulative_distribution']
        ax3 = axes[1, 0]
        ax3.plot(cumulative_data['diameters'], cumulative_data['cumulative'],
                 'b-', linewidth=2, color='darkblue')
        ax3.set_xlabel('孔隙直径 (单位)', fontsize=12)
        ax3.set_ylabel('累积概率', fontsize=12)
        ax3.set_title('孔隙直径累积分布函数', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_xscale('log')

        ax4 = axes[1, 1]
        ax4.axis('off')
        summary_text = (
            f"孔隙统计摘要\n"
            f"================\n"
            f"孔隙总数: {stats['total_pores']:,}\n"
            f"总孔隙体积: {stats['total_pore_volume']:.2f} 体素\n\n"
            f"直径统计:\n"
            f"  平均直径: {stats['avg_pore_diameter']:.2f}\n"
            f"  体积加权平均直径: {stats['volume_weighted_diameter']:.2f}\n"
            f"  最小直径: {stats['min_pore_diameter']:.2f}\n"
            f"  最大直径: {stats['max_pore_diameter']:.2f}\n\n"
            f"分位数:\n"
            f"  中位数 (P50): {stats['diameter_percentiles'][2]:.2f}\n"
            f"  P75: {stats['diameter_percentiles'][3]:.2f}\n"
            f"  P90: {stats['diameter_percentiles'][4]:.2f}\n"
            f"  P95: {stats['diameter_percentiles'][5]:.2f}\n"
            f"  P99: {stats['diameter_percentiles'][6]:.2f}"
        )

        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                 fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"\n孔隙尺寸分布图已保存至: {save_path}")

        plt.show()


# 辅助函数：递归转换numpy类型为Python原生类型
def convert_numpy_to_python(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(i) for i in obj]
    else:
        return obj


if __name__ == "__main__":
    def process_single_sample_complete():
        print("=== 单个样本三维孔隙分析 ===")

        # 初始化重建器
        reconstructor = PoreReconstruction3D(voxel_size=(1.0, 1.0, 1.0))

        # 加载图像
        reconstructor.load_binary_images(
            folder_path='C:\\Users\\dell\\PycharmProjects\\Mask2AM\\img\\pore_filled\\HL1\\',
            file_pattern="*.png"
        )

        # 三维重建
        reconstructor.marching_cubes(level=128)

        # 孔隙率计算
        porosity = reconstructor.calculate_porosity()

        # 孔隙尺寸分析
        pore_stats = reconstructor.analyze_pore_size_distribution()

        # 绘制分布图
        reconstructor.plot_pore_size_distribution(save_path='results/pore_distribution.png')

        # 可视化原始Marching Cubes结果
        reconstructor.visualize_3d(save_path='results/3d_pores.html')

        # 导出PLY网格
        reconstructor.export_mesh('results/pore_structure.ply', format='ply')

        # 保存结果
        result_dir = 'results'
        os.makedirs(result_dir, exist_ok=True)

        print("\n=== 保存分析结果 ===")
        with open(os.path.join(result_dir, 'porosity_results.json'), 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_to_python(porosity), f, indent=2, ensure_ascii=False)

        print("\n=== 保存统计信息 ===")
        stats_to_save = {
            k: v for k, v in pore_stats.items()
            if k not in ['pore_volume_distribution', 'pore_diameter_distribution',
                         'diameter_histogram', 'volume_histogram', 'cumulative_distribution']
        }
        stats_to_save['pore_volume_distribution_shape'] = len(pore_stats['pore_volume_distribution'])
        stats_to_save['pore_diameter_distribution_shape'] = len(pore_stats['pore_diameter_distribution'])
        stats_to_save = convert_numpy_to_python(stats_to_save)

        with open(os.path.join(result_dir, 'pore_statistics.json'), 'w', encoding='utf-8') as f:
            json.dump(stats_to_save, f, indent=2, ensure_ascii=False)

        # 保存分布数据
        np.savez_compressed(os.path.join(result_dir, 'pore_distributions.npz'),
                            pore_volumes=pore_stats['pore_volume_distribution'],
                            pore_diameters=pore_stats['pore_diameter_distribution'])

        print("分析完成！所有结果已保存到 results/ 目录")
        return reconstructor, porosity, pore_stats


    # 运行分析
    reconstructor, porosity, pore_stats = process_single_sample_complete()