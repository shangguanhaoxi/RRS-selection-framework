import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage.io import imsave


# -----------------------------
# Halton 序列生成函数（新增）
# -----------------------------
def halton_sequence(index, base):
    result = 0.0
    f = 1.0
    i = index
    while i > 0:
        f /= base
        result += f * (i % base)
        i = i // base
    return result


def generate_uniform_centers_halton(L, num_points):
    """
    使用 Halton 序列生成 num_points 个在 [0, L]×[0, L] 上的准均匀分布中心点。
    每个点对应一张独立图像，无需防重叠。
    """
    centers = []
    for i in range(num_points):
        x = L * halton_sequence(i, 2)  # base=2
        y = L * halton_sequence(i, 3)  # base=3
        centers.append((x, y))
    return centers


# -----------------------------
# 生成高斯凸起图像函数 (固定分辨率)
# -----------------------------
def generate_gaussian_bump_fixed_resolution(h, sigma, L, resolution_ratio, center=None):
    """
    在 LxL 的区域内生成一个高斯凸起图像。

    参数:
    h: 高斯凸起的最大高度。
    sigma: 高斯凸起的标准差。
    L: 区域的边长。
    resolution_ratio: 图像的分辨率 (resolution_ratio x resolution_ratio)。
    center: 凸起的中心 (x0, y0)。如果为 None，则随机生成。

    返回:
    Z: 生成的高斯图像 (resolution_ratio x resolution_ratio numpy array)。
    center: 实际使用的中心坐标 (x0, y0)。
    """
    # 创建采样网格
    x = np.linspace(0, L, resolution_ratio)
    y = np.linspace(0, L, resolution_ratio)
    X, Y = np.meshgrid(x, y)

    # 设置凸起中心
    if center is None:
        x0 = np.random.uniform(0, L)
        y0 = np.random.uniform(0, L)
        center = (x0, y0)
    else:
        x0, y0 = center

    # 计算高斯高度
    distances_squared = (X - x0) ** 2 + (Y - y0) ** 2
    Z = h * np.exp(-distances_squared / (2 * sigma ** 2))

    # 可选：应用掩码以限制凸起范围（例如，只保留 3*sigma 内的值）
    # mask = distances_squared <= (3 * sigma) ** 2
    # Z[~mask] = 0

    return Z, center


# -----------------------------
# 主函数：生成并保存图像
# -----------------------------
def generate_and_save_images(
        h_range=(4.0, 6.0),
        sigma_range=(5.0, 7.0),
        L=50.0,
        num_images=100,
        resolution_ratio=150,
        output_dir='./generated_images'
):
    """
    生成指定数量的单高斯凸起图像并保存。

    参数:
    h_range: 高度 h 的随机范围 (min, max)。
    sigma_range: 标准差 sigma 的随机范围 (min, max)。
    L: 图像区域的边长。
    num_images: 要生成的图像总数。
    resolution_ratio: 每张图像的分辨率。
    output_dir: 保存图像和Excel文件的目录。
    """
    os.makedirs(output_dir, exist_ok=True)

    all_centers_x = []
    all_centers_y = []
    all_max_heights = []
    all_sigmas = []
    all_filenames = []
    all_heights = []  # 新增：存储所有h值

    # ✅ 关键修改1：提前生成所有中心点（Halton 序列）
    centers = generate_uniform_centers_halton(L, num_images)

    for i in range(num_images):
        # 1. 为每张图随机生成参数（保持原样）
        h = np.random.uniform(h_range[0], h_range[1])
        sigma = np.random.uniform(sigma_range[0], sigma_range[1])

        # ✅ 关键修改2：传入预生成的 Halton 中心点，不再随机
        Z, center = generate_gaussian_bump_fixed_resolution(
            h=h,
            sigma=sigma,
            L=L,
            resolution_ratio=resolution_ratio,
            center=centers[i]  # ✅ 使用 Halton 序列生成的中心
        )

        # 3. 保存图像（保持原样）
        filename = f"{i + 1}.tiff"
        filepath = os.path.join(output_dir, filename)
        imsave(filepath, Z.astype(np.float32), check_contrast=False)

        # 4. 记录信息（保持原样）
        all_centers_x.append(center[0])
        all_centers_y.append(center[1])
        all_max_heights.append(np.max(Z))
        all_sigmas.append(sigma)
        all_filenames.append(filename)
        all_heights.append(h)  # 新增：记录h值

        print(
            f"Saved image {i + 1}/{num_images}: {filepath} "
            f"(Center: {center[0]:.2f}, {center[1]:.2f}; Height: {h:.2f}, Sigma: {sigma:.2f})"
        )

    # 5. 保存信息到Excel文件（新增h值列）
    df = pd.DataFrame({
        'Filename': all_filenames,
        'Center_X': all_centers_x,
        'Center_Y': all_centers_y,
        'Max_Height': all_max_heights,
        'Sigma_Used': all_sigmas,
        'Height_h': all_heights  # 新增：h值列
    })

    excel_filename = os.path.join(output_dir, 'image_parameters.xlsx')
    df.to_excel(excel_filename, index=False)
    print(f"\nSummary saved to {excel_filename}")

    # 6. 可视化：显示所有中心点的分布（保持原样）
    plt.figure(figsize=(8, 8))
    plt.scatter(all_centers_x, all_centers_y, c='blue', alpha=0.6, edgecolors='k')
    plt.xlim(0, L)
    plt.ylim(0, L)
    plt.title('Distribution of Gaussian Bump Centers\n(Using Halton Sequence for Uniformity)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plot_filepath = os.path.join(output_dir, 'center_distribution.png')
    plt.savefig(plot_filepath, dpi=200)
    plt.show()
    plt.close()
    print(f"Center distribution plot saved to {plot_filepath}")

    # 7. 打印统计信息（新增h的平均值）
    mean_sigma = np.mean(all_sigmas)
    mean_height = np.mean(all_heights)  # 新增：计算h的平均值
    target_sigma_median = np.mean(sigma_range)
    target_height_median = np.mean(h_range)  # 新增：计算h范围的平均值

    print(f"\n--- Statistics ---")
    print(f"Target sigma range: {sigma_range}")
    print(f"Mean of sigmas used: {mean_sigma:.4f}")
    print(f"Target sigma range mean: {target_sigma_median:.4f}")
    print(f"Target height range: {h_range}")
    print(f"Mean of heights used: {mean_height:.4f}")  # 新增：显示h的平均值
    print(f"Target height range mean: {target_height_median:.4f}")  # 新增：显示h范围的平均值
    print(f"Number of images generated: {num_images}")
    print(f"Image resolution: {resolution_ratio} x {resolution_ratio}")
    print(f"Region size (L): {L}")


# -----------------------------
# 主程序入口
# -----------------------------
if __name__ == "__main__":
    generate_and_save_images(
        h_range=(4, 5),
        sigma_range=(5, 7),
        L=50,
        num_images=2000,
        resolution_ratio=150,
        output_dir='/root/autodl-tmp/test_1'
    )