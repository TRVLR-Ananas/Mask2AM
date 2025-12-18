import open3d as o3d
import numpy as np
from scipy.linalg import solve
import warnings

warnings.filterwarnings('ignore')


class EllipsoidParams:
    def __init__(self):
        self.center = np.zeros(3)
        self.semi_axes = np.zeros(3)
        self.rotation = np.eye(3)
        self.volume = 0.0


def fit_ellipsoid(points):
    """适配大尺度点云的椭球拟合（增加点数阈值判断）"""
    params = EllipsoidParams()
    n = points.shape[0]

    # 拟合椭球至少需要9个点，大孔隙可抽样拟合（加速）
    if n < 9:
        print(f"错误：点云数量不足（{n}<9），无法拟合椭球！")
        return None
    # 对超多点的孔隙抽样拟合（避免计算量过大）
    sample_n = min(n, 10000)  # 最多抽样1万个点
    if n > sample_n:
        idx = np.random.choice(n, sample_n, replace=False)
        points_sample = points[idx]
    else:
        points_sample = points

    x = points_sample[:, 0]
    y = points_sample[:, 1]
    z = points_sample[:, 2]

    A = np.zeros((sample_n, 9))
    A[:, 0] = x * x
    A[:, 1] = y * y
    A[:, 2] = z * z
    A[:, 3] = x * y
    A[:, 4] = x * z
    A[:, 5] = y * z
    A[:, 6] = x
    A[:, 7] = y
    A[:, 8] = z

    b = -np.ones(sample_n)

    try:
        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
    except:
        print("错误：最小二乘求解系数失败！")
        return None

    Q = np.array([
        [coeffs[0], coeffs[3] / 2, coeffs[4] / 2],
        [coeffs[3] / 2, coeffs[1], coeffs[5] / 2],
        [coeffs[4] / 2, coeffs[5] / 2, coeffs[2]]
    ])
    L = np.array([coeffs[6] / 2, coeffs[7] / 2, coeffs[8] / 2])

    try:
        params.center = solve(Q, -L, assume_a='pos')
    except:
        print("错误：求解椭球中心失败（矩阵奇异）！")
        return None

    c = params.center
    cTcQc = c.T @ Q @ c
    K = cTcQc + L.T @ c + 1.0
    Q_norm = -Q / K

    try:
        eig_vals, eig_vecs = np.linalg.eig(Q_norm)
    except:
        print("错误：特征值分解失败！")
        return None

    if np.any(eig_vals <= 1e-6):
        print(f"错误：拟合结果非椭球（特征值={eig_vals}）！")
        return None

    semi_axes = 1 / np.sqrt(eig_vals)
    sorted_idx = np.argsort(semi_axes)[::-1]
    params.semi_axes = semi_axes[sorted_idx]
    params.rotation = eig_vecs[:, sorted_idx]
    params.volume = (4 / 3) * np.pi * np.prod(params.semi_axes)

    # 可选：计算拟合误差（抽样计算）
    points_local = (points_sample - params.center) @ params.rotation
    points_unit = points_local / params.semi_axes
    distances = np.abs(np.linalg.norm(points_unit, axis=1) - 1)
    print(f"平均拟合误差：{np.mean(distances):.6f}")

    return params


def main(pcd_path):
    # 1. 读取并清理点云
    print(f"读取点云文件：{pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    if len(pcd.points) == 0:
        print("错误：读取的点云为空！")
        return

    # 清理NaN/Inf点
    points = np.asarray(pcd.points)
    mask = np.isfinite(points).all(axis=1)
    pcd = pcd.select_by_index(np.where(mask)[0])
    points = np.asarray(pcd.points)
    print(f"清理无效点后点数：{len(points)}")

    # 打印点云尺度（关键！坐标范围0~673 → 单位是毫米）
    print("\n点云坐标范围：")
    print(f"X: {np.min(points[:, 0]):.6f} ~ {np.max(points[:, 0]):.6f}")
    print(f"Y: {np.min(points[:, 1]):.6f} ~ {np.max(points[:, 1]):.6f}")
    print(f"Z: {np.min(points[:, 2]):.6f} ~ {np.max(points[:, 2]):.6f}")

    # 2. 关键优化：体素下采样（减少点云数量，避免卡死）
    # voxel_size=1 → 1毫米体素，千万级点云可降到几万/几十万点
    voxel_size = 1.0  # 可根据需求调整（0.5/2.0）
    print(f"\n执行体素下采样（voxel_size={voxel_size}毫米）...")
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"下采样后点数：{len(pcd_down.points)}")

    # 3. 宽松的统计滤波（仅过滤极端噪声）
    print("\n执行统计滤波...")
    cl, ind = pcd_down.remove_statistical_outlier(
        nb_neighbors=50,  # 近邻数适配下采样后的点云
        std_ratio=2.0  # 宽松阈值，避免过滤有效点
    )
    pcd_filtered = pcd_down.select_by_index(ind)
    print(f"滤波后点数：{len(pcd_filtered.points)}")

    # 4. 高效聚类：使用Open3D的DBSCAN（适配毫米单位）
    # 关键：tolerance=2 → 2毫米（坐标单位是毫米，无需转米！）
    eps = 2.0  # 聚类容忍度（毫米），可调整为1/3/5
    min_points = 10  # 最小聚类点数（适配下采样后的点云）
    print(f"\n执行DBSCAN聚类（eps={eps}毫米, min_points={min_points}）...")
    labels = np.array(pcd_filtered.cluster_dbscan(
        eps=eps,
        min_points=min_points,
        print_progress=True  # 显示进度条
    ))

    # 调试聚类结果
    unique_labels = np.unique(labels)
    valid_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    print(f"\n聚类结果：")
    print(f"所有标签：{unique_labels[:10]}...（共{len(unique_labels)}个）")
    print(f"噪声点数量：{np.sum(labels == -1)}")
    print(f"有效孔隙聚类数量：{valid_clusters}")

    if valid_clusters == 0:
        print("\n⚠️ 未检测到有效聚类！尝试：")
        print(f"1. 增大eps（当前={eps}）到3/5毫米")
        print(f"2. 减小min_points（当前={min_points}）到5/3")
        print(f"3. 减小voxel_size（当前={voxel_size}）到0.5")
        return

    # 5. 拟合椭球（仅处理前N个聚类，避免耗时过久）
    max_label = labels.max()
    print(f"\n检测到{max_label + 1}个孔隙，开始拟合椭球（仅处理前20个）...")
    process_num = min(20, max_label + 1)  # 限制处理数量，可调整

    for pore_id in range(process_num):
        # 提取单个孔隙的点云
        pore_indices = np.where(labels == pore_id)[0]
        if len(pore_indices) == 0:
            continue
        pore_pcd = pcd_filtered.select_by_index(pore_indices)
        pore_points = np.asarray(pore_pcd.points)

        print(f"\n================= 孔隙 {pore_id} =================")
        print(f"孔隙点云数量：{len(pore_points)}")

        # 拟合椭球
        ellipsoid_params = fit_ellipsoid(pore_points)
        if ellipsoid_params is None:
            print(f"孔隙 {pore_id} 椭球拟合失败！")
            continue

        # 输出椭球参数（单位：毫米）
        print(f"椭球中心 (x,y,z)：{ellipsoid_params.center.round(2)} 毫米")
        print(f"半轴长 (长,中,短)：{ellipsoid_params.semi_axes.round(2)} 毫米")
        print(f"旋转矩阵：\n{ellipsoid_params.rotation.round(4)}")
        print(f"椭球体积：{ellipsoid_params.volume:.2f} 立方毫米")


if __name__ == "__main__":
    # 替换为你的PCD文件路径
    PCD_FILE_PATH = r"C:\Users\dell\PycharmProjects\Mask2AM\pcd\pore_filled\HL1.pcd"
    main(PCD_FILE_PATH)
