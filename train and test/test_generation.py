import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage.io import imsave


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


def generate_gaussian_bump_fixed_resolution(h, sigma, L, resolution_ratio, center=None):

    x = np.linspace(0, L, resolution_ratio)
    y = np.linspace(0, L, resolution_ratio)
    X, Y = np.meshgrid(x, y)

    if center is None:
        x0 = np.random.uniform(0, L)
        y0 = np.random.uniform(0, L)
        center = (x0, y0)
    else:
        x0, y0 = center


    distances_squared = (X - x0) ** 2 + (Y - y0) ** 2
    Z = h * np.exp(-distances_squared / (2 * sigma ** 2))



    return Z, center


def generate_and_save_images(
        h_range=(4.0, 6.0),
        sigma_range=(5.0, 7.0),
        L=50.0,
        num_images=100,
        resolution_ratio=150,
        output_dir='./generated_images'
):
  
    os.makedirs(output_dir, exist_ok=True)

    all_centers_x = []
    all_centers_y = []
    all_max_heights = []
    all_sigmas = []
    all_filenames = []
    all_heights = [] 

    centers = generate_uniform_centers_halton(L, num_images)

    for i in range(num_images):
  
        h = np.random.uniform(h_range[0], h_range[1])
        sigma = np.random.uniform(sigma_range[0], sigma_range[1])

 
        Z, center = generate_gaussian_bump_fixed_resolution(
            h=h,
            sigma=sigma,
            L=L,
            resolution_ratio=resolution_ratio,
            center=centers[i] 
        )

    
        filename = f"{i + 1}.tiff"
        filepath = os.path.join(output_dir, filename)
        imsave(filepath, Z.astype(np.float32), check_contrast=False)

    
        all_centers_x.append(center[0])
        all_centers_y.append(center[1])
        all_max_heights.append(np.max(Z))
        all_sigmas.append(sigma)
        all_filenames.append(filename)
        all_heights.append(h) 

        print(
            f"Saved image {i + 1}/{num_images}: {filepath} "
            f"(Center: {center[0]:.2f}, {center[1]:.2f}; Height: {h:.2f}, Sigma: {sigma:.2f})"
        )


    df = pd.DataFrame({
        'Filename': all_filenames,
        'Center_X': all_centers_x,
        'Center_Y': all_centers_y,
        'Max_Height': all_max_heights,
        'Sigma_Used': all_sigmas,
        'Height_h': all_heights  
    })

    excel_filename = os.path.join(output_dir, 'image_parameters.xlsx')
    df.to_excel(excel_filename, index=False)
    print(f"\nSummary saved to {excel_filename}")

  
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

 
    mean_sigma = np.mean(all_sigmas)
    mean_height = np.mean(all_heights) 
    target_sigma_median = np.mean(sigma_range)
    target_height_median = np.mean(h_range)

    print(f"\n--- Statistics ---")
    print(f"Target sigma range: {sigma_range}")
    print(f"Mean of sigmas used: {mean_sigma:.4f}")
    print(f"Target sigma range mean: {target_sigma_median:.4f}")
    print(f"Target height range: {h_range}")
    print(f"Mean of heights used: {mean_height:.4f}") 
    print(f"Target height range mean: {target_height_median:.4f}") 
    print(f"Number of images generated: {num_images}")
    print(f"Image resolution: {resolution_ratio} x {resolution_ratio}")
    print(f"Region size (L): {L}")


if __name__ == "__main__":
    generate_and_save_images(
        h_range=(4, 5),
        sigma_range=(5, 7),
        L=50,
        num_images=2000,
        resolution_ratio=150,
        output_dir='/root/autodl-tmp/test_1'

    )
