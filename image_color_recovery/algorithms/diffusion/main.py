"""
Example script for PDE-based colorization. 
Uses Google Colab file upload and a basic grid search for sigma.
"""

import cv2
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from skimage import color
from tqdm import trange
from google.colab import files

from .solver import pde_colorize_3ch_with_error


def downsample_rgb(img, scale):
    """
    Downsample an RGB image using INTER_AREA interpolation.

    Parameters
    ----------
    img : np.ndarray, shape (H, W, 3), float32 in [0,1]
        Input image.
    scale : float
        Factor by which to downsample.

    Returns
    -------
    np.ndarray
        Downsampled image in [0,1].
    """
    new_w = int(img.shape[1] * scale)
    new_h = int(img.shape[0] * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def run_pde_demo():
    """
    Demonstration of the PDE-based diffusion colorization method.
    """
    print("Please upload a color image (used as ground truth).")
    uploaded = files.upload()
    base_path = list(uploaded.keys())[0]
    img_bgr = cv2.imread(base_path)
    if img_bgr is None:
        raise ValueError("Could not read the image.")

    # Convert to float32 in [0,1]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Downsample the image for speed
    scale_factor = 0.5
    img_rgb_small = downsample_rgb(img_rgb, scale_factor)
    height, width, _ = img_rgb_small.shape

    # Convert to grayscale in [0,1]
    gray_cpu = color.rgb2gray(img_rgb_small).astype(np.float32)
    gray_gpu = cp.asarray(gray_cpu)

    # Ground truth 3-channel image in [0,1]
    original_3ch_cpu = img_rgb_small
    original_3ch_gpu = cp.asarray(original_3ch_cpu)

    # Create known mask and values
    pct_known = 1
    num_known = int((pct_known / 100.0) * height * width)
    all_indices = np.arange(height * width)
    chosen = np.random.choice(all_indices, size=num_known, replace=False)

    known_mask_cpu = np.zeros((height, width), dtype=bool)
    known_values_3ch_cpu = np.zeros((height, width, 3), dtype=np.float32)
    known_mask_cpu.ravel()[chosen] = True
    known_values_3ch_cpu[known_mask_cpu] = original_3ch_cpu[known_mask_cpu]

    known_mask_gpu = cp.asarray(known_mask_cpu)
    known_values_3ch_gpu = cp.asarray(known_values_3ch_cpu)

    # PDE parameters
    sigma_val = 0.1
    max_iters = 35

    # Run PDE colorization
    recovered_3ch_gpu, error_values = pde_colorize_3ch_with_error(
        gray_gpu,
        known_mask_gpu,
        known_values_3ch_gpu,
        original_3ch_gpu,
        sigma=sigma_val,
        max_iters=max_iters
    )
    recovered_3ch_cpu = cp.asnumpy(recovered_3ch_gpu).clip(0, 1)

    # Plot error vs. iteration
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(error_values) + 1), error_values,
             marker='o', linestyle='none')
    plt.xlabel("Iteration")
    plt.ylabel("Error (MSE)")
    plt.title(f"Diffusion-Propagation Error vs. Iterations ({pct_known}%)")
    plt.grid(True)
    plt.show()

    # Calculate and print error improvement
    initial_error = error_values[0]
    final_error = error_values[-1]
    if initial_error > 1e-12:
        perc_improvement = ((initial_error - final_error) / initial_error) * 100
    else:
        perc_improvement = 0.0

    print(f"Initial Error: {initial_error:.6f}")
    print(f"Final Error:   {final_error:.6f}")
    print(f"Percentage Improvement: {perc_improvement:.2f}%")

    # Grid search for sigma
    sigma_candidates = [0.05, 0.1, 0.2, 1.0, 2.0, 3.0, 5.0]
    best_sigma = None
    best_error = float('inf')
    best_recovered_gpu = None
    max_iters_grid = 30

    for s_val in sigma_candidates:
        rec_gpu, err_list = pde_colorize_3ch_with_error(
            gray_gpu,
            known_mask_gpu,
            known_values_3ch_gpu,
            original_3ch_gpu,
            sigma=s_val,
            max_iters=max_iters_grid
        )
        final_mse = err_list[-1]
        print(f"Sigma = {s_val:.3f}, Final MSE = {final_mse:.6f}")
        if final_mse < best_error:
            best_error = final_mse
            best_sigma = s_val
            best_recovered_gpu = rec_gpu

    print(f"\nOptimized sigma = {best_sigma:.3f} with MSE = {best_error:.6f}")
    best_recovered_cpu = cp.asnumpy(best_recovered_gpu).clip(0, 1)

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    # Original
    axs[0].imshow(img_rgb_small)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    # Recovered (Initial)
    axs[1].imshow(recovered_3ch_cpu)
    axs[1].set_title(f"Recovered (Initial) {pct_known}%")
    axs[1].set_xlabel(f"sigma={sigma_val:.2f}", fontsize=12, labelpad=10)
    axs[1].axis("off")

    # Recovered (Optimized)
    axs[2].imshow(best_recovered_cpu)
    axs[2].set_title(f"Recovered (Optimized) {pct_known}%")
    axs[2].set_xlabel(f"sigma={best_sigma:.2f}", fontsize=12, labelpad=10)
    axs[2].axis("off")

    plt.show()


if __name__ == "__main__":
    run_pde_demo()
