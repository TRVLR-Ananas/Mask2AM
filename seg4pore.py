from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, measure
import os
from pathlib import Path
from typing import Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')


class PoreSegmentation:

    def __init__(self, image_path: str = None):
        self.image_path = image_path
        self.original_image = None
        self.processed_image = None
        self.binary_mask = None
        self.filled_mask = None  # 填充后的掩码
        self.edge_mask = None    # 边缘掩码
        self.threshold_value = None

        if image_path:
            self.load_image(image_path)

    def load_image(self, image_path: str):
        self.image_path = image_path
        self.original_image = io.imread(image_path)

        if len(self.original_image.shape) == 3:
            self.original_image = np.dot(self.original_image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

        self.original_image = cv2.normalize(self.original_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        print(f"图像尺寸: {self.original_image.shape}")
        return self.original_image

    def preprocess_image(self, denoise=True, gaussian_sigma=1, median_size=3):

        if self.original_image is None:
            raise ValueError("请先加载图像")

        self.processed_image = self.original_image.copy()

        if denoise:
            # 高斯滤波去噪
            kernel_size = int(2 * np.ceil(2 * gaussian_sigma) + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            self.processed_image = cv2.GaussianBlur(self.processed_image,
                                                    (kernel_size, kernel_size),
                                                    gaussian_sigma)

            # 中值滤波去除椒盐噪声
            if median_size % 2 == 0:
                median_size += 1
            self.processed_image = cv2.medianBlur(self.processed_image, median_size)

        return self.processed_image

    def otsu_thresholding(self) -> Tuple[np.ndarray, float]:
        if self.processed_image is None:
            self.preprocess_image()

        threshold, binary_mask = cv2.threshold(
            self.processed_image,
            0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        self.binary_mask = binary_mask
        self.threshold_value = threshold

        print(f"Otsu阈值: {threshold:.2f}")
        return binary_mask, threshold

    def adaptive_thresholding(self, block_size=11, C=2, method='gaussian'):

        if self.processed_image is None:
            self.preprocess_image()

        if block_size % 2 == 0:
            block_size += 1

        # 选择自适应方法
        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == 'gaussian' else cv2.ADAPTIVE_THRESH_MEAN_C

        binary_mask = cv2.adaptiveThreshold(
            self.processed_image,
            255,
            adaptive_method,
            cv2.THRESH_BINARY_INV,
            block_size,
            C
        )

        self.binary_mask = binary_mask
        self.threshold_value = None

        print(f"自适应阈值分割完成，方法: {method}, 块大小: {block_size}, C: {C}")
        return binary_mask

    def manual_thresholding(self, threshold: int):

        if self.processed_image is None:
            self.preprocess_image()

        _, binary_mask = cv2.threshold(
            self.processed_image,
            threshold, 255,
            cv2.THRESH_BINARY_INV
        )

        self.binary_mask = binary_mask
        self.threshold_value = threshold

        print(f"手动阈值: {threshold}")
        return binary_mask

    def multi_otsu_thresholding(self, classes: int = 3):

        if self.processed_image is None:
            self.preprocess_image()

        # 使用skimage的多级Otsu
        from skimage.filters import threshold_multiotsu

        thresholds = threshold_multiotsu(self.processed_image, classes=classes)

        # 根据阈值分割图像
        regions = np.digitize(self.processed_image, bins=thresholds)

        binary_mask = np.uint8(regions == 0) * 255

        self.binary_mask = binary_mask
        self.threshold_value = thresholds[0]

        print(f"多级Otsu分割阈值: {thresholds}")
        return binary_mask

    def post_processing(self,
                        remove_small_objects=True,
                        min_size=50,
                        fill_holes=True,
                        morph_operations=True):

        if self.binary_mask is None:
            raise ValueError("请先进行阈值分割")

        processed_mask = self.binary_mask.copy()

        # 形态学操作
        if morph_operations:
            # 开运算去除小噪声（先腐蚀后膨胀）
            kernel = np.ones((3, 3), np.uint8)
            processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)

            # 闭运算填充小孔（先膨胀后腐蚀）
            processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel)

        # 移除小物体
        if remove_small_objects:
            # 寻找连通区域
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                processed_mask, connectivity=8
            )

            # 创建新掩码
            new_mask = np.zeros_like(processed_mask)

            # 保留大于最小尺寸的区域
            for i in range(1, num_labels):  # 跳过背景
                if stats[i, cv2.CC_STAT_AREA] >= min_size:
                    new_mask[labels == i] = 255

            processed_mask = new_mask

        # 填充孔洞
        if fill_holes:
            # 使用floodFill从左上角开始填充背景，然后取反
            h, w = processed_mask.shape
            mask = np.zeros((h + 2, w + 2), np.uint8)
            # 复制掩码用于填充
            flood_mask = processed_mask.copy()
            # 从(0,0)位置填充背景（黑色区域）
            cv2.floodFill(flood_mask, mask, (0, 0), 255)
            # 取反得到孔洞区域，再合并到原掩码
            flood_mask_inv = cv2.bitwise_not(flood_mask)
            processed_mask = processed_mask | flood_mask_inv

        self.binary_mask = processed_mask
        return processed_mask

    def extract_pore_edges(self, edge_thickness: int = 1):
        """
        提取孔隙边缘
        返回二值图像：白色边缘，黑色背景
        """
        if self.binary_mask is None:
            raise ValueError("请先进行分割和后处理")

        # 创建全黑的边缘图像
        self.edge_mask = np.zeros_like(self.binary_mask)

        # 找到所有轮廓（外部轮廓）
        contours, _ = cv2.findContours(
            self.binary_mask,
            cv2.RETR_EXTERNAL,  # 只检测外部轮廓
            cv2.CHAIN_APPROX_SIMPLE
        )

        # 在边缘图像上绘制白色轮廓
        cv2.drawContours(
            self.edge_mask,
            contours,
            -1,  # 绘制所有轮廓
            255,  # 白色
            edge_thickness  # 边缘厚度
        )

        print(f"找到 {len(contours)} 个封闭孔隙")
        return self.edge_mask

    def calculate_porosity(self) -> float:

        if self.binary_mask is None:
            raise ValueError("请先进行分割")

        # 计算白色像素（孔隙）的比例
        total_pixels = self.binary_mask.size
        pore_pixels = np.sum(self.binary_mask == 255)

        porosity = (pore_pixels / total_pixels) * 100

        print(f"孔隙率: {porosity:.2f}%")
        return porosity

    def analyze_pores(self):

        if self.binary_mask is None:
            raise ValueError("请先进行分割")

        # 寻找连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            self.binary_mask, connectivity=8
        )

        properties = []

        print(f"\n孔隙分析结果:")
        print(f"总孔隙数量: {num_labels - 1}")

        # 为每个孔隙计算属性
        for i in range(1, num_labels):  # 跳过背景
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 0:
                # 计算等效直径
                equivalent_diameter = 2 * np.sqrt(area / np.pi)

                # 获取边界框
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_TOP + 3]

                properties.append({
                    'id': i,
                    'area': area,
                    'equivalent_diameter': equivalent_diameter,
                    'centroid': centroids[i],
                    'bounding_box': (x, y, w, h)
                })

        return properties

    def visualize(self,
                  show_original: bool = True,
                  show_binary: bool = True,
                  show_edges: bool = True,
                  show_overlay: bool = True):
        """可视化所有步骤"""
        # 计算需要显示的图像数量
        num_plots = sum([show_original, show_binary, show_edges, show_overlay])
        if num_plots == 0:
            return

        fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4))
        if num_plots == 1:
            axes = [axes]

        idx = 0

        # 1. 原始图像
        if show_original and self.original_image is not None:
            axes[idx].imshow(self.original_image, cmap='gray')
            axes[idx].set_title('原始图像')
            axes[idx].axis('off')
            idx += 1

        # 2. 二值分割结果
        if show_binary and self.binary_mask is not None:
            axes[idx].imshow(self.binary_mask, cmap='gray')
            axes[idx].set_title('二值分割')
            axes[idx].axis('off')
            idx += 1

        # 3. 边缘图像
        if show_edges and self.edge_mask is not None:
            axes[idx].imshow(self.edge_mask, cmap='gray')
            axes[idx].set_title('孔隙边缘')
            axes[idx].axis('off')
            idx += 1

        # 4. 边缘叠加显示
        if show_overlay and self.original_image is not None and self.edge_mask is not None:
            # 创建叠加图像
            if len(self.original_image.shape) == 2:
                overlay = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2RGB)
            else:
                overlay = self.original_image.copy()

            # 找到边缘点并标记为红色
            edge_points = np.where(self.edge_mask == 255)
            overlay[edge_points] = [255, 0, 0]  # 红色

            axes[idx].imshow(overlay)
            axes[idx].set_title('边缘叠加')
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()

    def save_results(self,
                     output_dir: str = 'results',
                     save_binary: bool = True,
                     save_edges: bool = True,
                     save_overlay: bool = False):

        if self.binary_mask is None:
            raise ValueError("请先进行处理")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(self.image_path).stem

        # 保存二值图像
        if save_binary:
            binary_path = os.path.join(output_dir, f'{base_name}_binary.png')
            cv2.imwrite(binary_path, self.binary_mask)
            print(f"二值图像保存至: {binary_path}")

        # 保存边缘图像（核心输出）
        if save_edges:
            if self.edge_mask is None:
                self.extract_pore_edges()

            edge_path = os.path.join(output_dir, f'{base_name}_edges.png')
            cv2.imwrite(edge_path, self.edge_mask)
            print(f"边缘图像保存至: {edge_path}")

        # 保存叠加图像
        if save_overlay and self.original_image is not None and self.edge_mask is not None:
            if len(self.original_image.shape) == 2:
                overlay = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2RGB)
            else:
                overlay = self.original_image.copy()

            # 标记边缘为红色
            edge_points = np.where(self.edge_mask == 255)
            overlay[edge_points] = [255, 0, 0]

            overlay_path = os.path.join(output_dir, f'{base_name}_overlay.png')
            cv2.imwrite(overlay_path, overlay)
            print(f"叠加图像保存至: {overlay_path}")


def batch_process_images(input_dir: str,
                         output_dir: str = 'batch_results',
                         method: str = 'otsu',
                         save_type: str = 'filled',
                         **kwargs):

    os.makedirs(output_dir, exist_ok=True)

    # 获取所有图像文件
    image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
    image_files = []

    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
        image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))

    print(f"找到 {len(image_files)} 个图像文件")

    results = []

    for img_path in image_files:
        print(f"\n处理图像: {img_path.name}")

        try:
            # 创建分割器实例
            seg = PoreSegmentation(str(img_path))

            # 预处理
            seg.preprocess_image(denoise=False, gaussian_sigma=1, median_size=3)

            # 选择分割方法
            if method == 'otsu':
                seg.otsu_thresholding()
            elif method == 'adaptive':
                block_size = kwargs.get('block_size', 11)
                C = kwargs.get('C', 2)
                method_type = kwargs.get('method_type', 'gaussian')
                seg.adaptive_thresholding(block_size, C, method_type)
            elif method == 'manual':
                threshold = kwargs.get('threshold', 127)
                seg.manual_thresholding(threshold)
            elif method == 'multi_otsu':
                classes = kwargs.get('classes', 3)
                seg.multi_otsu_thresholding(classes)

            # 后处理
            seg.post_processing(
                remove_small_objects=False,
                min_size=5,  # 如果孔隙很小，可减小这个值，比如10
                fill_holes=False,
                morph_operations=True
            )

            # 计算孔隙率
            porosity = seg.calculate_porosity()

            # 提取边缘
            edges = seg.extract_pore_edges(edge_thickness=1)

            # 保存结果
            if save_type == 'edge':
                seg.save_results(output_dir, save_binary = False, save_edges = True)
            else:
                seg.save_results(output_dir, save_binary = True, save_edges = False)

            results.append({
                'filename': img_path.name,
                'porosity': porosity,
                'threshold': seg.threshold_value
            })

        except Exception as e:
            print(f"处理 {img_path.name} 时出错: {e}")

    # 保存汇总结果
    if results:
        summary_path = os.path.join(output_dir, 'summary.csv')
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(summary_path, index=False)
        print(f"\n处理完成！汇总结果已保存至: {summary_path}")


if __name__ == "__main__":
    def single_image():
        seg = PoreSegmentation('C:\\Users\\dell\\PycharmProjects\\Mask2AM\\img\\resource\\HL1\\HL1004.png')
        # 预处理
        seg.preprocess_image(denoise=False, gaussian_sigma=1, median_size=3)
        seg.otsu_thresholding()
        # 后处理
        seg.post_processing(
            remove_small_objects=False,
            min_size=5,  # 如果孔隙很小，可减小这个值，比如10
            fill_holes=False,
            morph_operations=True
        )

        # 计算孔隙率
        porosity = seg.calculate_porosity()

        # 边缘提取
        edges = seg.extract_pore_edges(edge_thickness=1)

        # 分析孔隙特征
        properties = seg.analyze_pores()

        # 可视化结果
        seg.visualize(
            show_original=True,
            show_binary=True,
            show_edges=True,
            show_overlay=True
        )

        # 保存结果
        # seg.save_results('results')

        return seg
    seg = single_image()

    # def batch_processing():
    #     batch_process_images(
    #         input_dir='C:\\Users\\dell\\PycharmProjects\\Mask2AM\\img\\resource\\S3\\',
    #         output_dir='C:\\Users\\dell\\PycharmProjects\\Mask2AM\\img\\pore_edge\\S3\\',
    #         method='otsu',  # 可选: 'otsu', 'adaptive', 'manual', 'multi_otsu'
    #         save_type='edge',
    #         # block_size=15,
    #         # C=2,
    #         # method_type='gaussian'
    #     )
    # batch_processing()