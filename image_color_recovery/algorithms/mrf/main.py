"""
Demo script for MRF-based colorization using partial color cues.
"""

import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from google.colab import files
from skimage.color import rgb2lab, lab2rgb
from tqdm import tqdm

from .solver import solve_channel_direct, solve_channel_iterative
from .utils import (
    compute_mrf_normalized_error,
    compute_mrf_normalized_error_channel
)


def run_mrf_demo():
    """
    1) Upload an image (RGB).
    2) Convert to Lab.
    3) Randomly mask partial color cues in a/b channels.
    4) Build & solve MRF system for each channel.
    5) Grid search over sigma, display best result.
    6) Optional iterative solver demonstration.
    """
    uploaded = files.upload()
    filename = list(uploaded.keys())[0]
    print("Using file:", filename)

    # Load image
    with io.BytesIO(uploaded[filename]) as f:
        img_pil = Image.open(f).convert('RGB')
    img_np = np.array(img_pil, dtype=np.float32) / 255.0
    height, width, _channels = img_np.shape
    print(f"Image shape: {height} x {width}")

    # Convert to Lab
    lab_img = rgb2lab(img_np)
    lightness_channel = lab_img[:, :, 0]
    a_gt = np.clip(lab_img[:, :, 1], -128, 128)
    b_gt = np.clip(lab_img[:, :, 2], -128, 128)

    # Simulate partial color cues
    known_percentage = 10
    mask = np.random.rand(height, width) < (known_percentage / 100.0)
    a_known = np.where(mask, a_gt, np.nan)
    b_known = np.where(mask, b_gt, np.nan)
    print(f"Known fraction ~ {mask.mean():.4f}")

    # Grid search over sigma
    sigma_list = np.linspace(1, 50, 10)
    errors = []

    print("\nPerforming grid search over sigma values...")
    for sigma_val in tqdm(sigma_list):
        a_temp = solve_channel_direct(lightness_channel, a_known, sigma_val)
        b_temp = solve_channel_direct(lightness_channel, b_known, sigma_val)
        nerr = compute_mrf_normalized_error(a_temp, b_temp, a_gt, b_gt)
        errors.append(nerr)

    errors = np.array(errors)
    opt_idx = np.argmin(errors)
    opt_sigma = sigma_list[opt_idx]
    print(f"\nOptimal sigma: {opt_sigma:.2f}, "
          f"Normalized Error={errors[opt_idx]:.6f}")

    # Recover images with original & optimal sigma
    orig_sigma = 5.0
    print(f"\nRecovering image with original sigma={orig_sigma}")
    a_rec_orig = solve_channel_direct(lightness_channel, a_known, orig_sigma)
    b_rec_orig = solve_channel_direct(lightness_channel, b_known, orig_sigma)
    nerr_orig = compute_mrf_normalized_error(a_rec_orig, b_rec_orig, a_gt, b_gt)
    print(f"Normalized Error (orig sigma) = {nerr_orig:.6f}")

    print(f"Recovering image with optimal sigma={opt_sigma:.2f}")
    a_rec_opt = solve_channel_direct(lightness_channel, a_known, opt_sigma)
    b_rec_opt = solve_channel_direct(lightness_channel, b_known, opt_sigma)
    nerr_opt = compute_mrf_normalized_error(a_rec_opt, b_rec_opt, a_gt, b_gt)
    print(f"Normalized Error (opt sigma)  = {nerr_opt:.6f}")

    # Convert to RGB
    lab_rec_orig = np.stack((lightness_channel, a_rec_orig, b_rec_orig), axis=2)
    rgb_rec_orig = lab2rgb(lab_rec_orig)
    lab_rec_opt = np.stack((lightness_channel, a_rec_opt, b_rec_opt), axis=2)
    rgb_rec_opt = lab2rgb(lab_rec_opt)

    # Display final results
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    # Original
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Recovered (Initial)
    axes[1].imshow(rgb_rec_orig)
    axes[1].set_title(f"Recovered MRF (Initial) {known_percentage}%")
    axes[1].set_xlabel(f"σ={orig_sigma}", fontsize=12, labelpad=10)
    axes[1].axis("off")

    # Recovered (Optimized)
    axes[2].imshow(rgb_rec_opt)
    axes[2].set_title(f"Recovered MRF (Optimized) {known_percentage}%")
    axes[2].set_xlabel(f"σ={opt_sigma:.2f}", fontsize=12, labelpad=10)
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    # Demonstrate iterative solver for the optimal sigma
    print("\n--- Demonstrate Iterative Solver for Optimal Sigma ---")
    a_cg, _ = iterative_demo(lightness_channel, a_known, a_gt, opt_sigma)
    b_cg, _ = iterative_demo(lightness_channel, b_known, b_gt, opt_sigma)
    nerr_cg = compute_mrf_normalized_error(a_cg, b_cg, a_gt, b_gt)
    print(f"Final CG Error (sigma={opt_sigma:.2f}) = {nerr_cg:.6f}")


def iterative_demo(lightness_channel, channel_known, channel_gt, sigma):
    """
    Demonstrate the iterative solver for one channel (a or b),
    tracking the normalized error at each iteration.
    """
    solutions_list = []
    channel_final, _info = solve_channel_iterative(
        lightness_channel, channel_known, sigma,
        tol=1e-6, maxiter=200, callback_list=solutions_list
    )

    # Compute normalized error at each iteration
    error_list = []
    for sol in solutions_list:
        sol_2d = sol.reshape(lightness_channel.shape)
        err = compute_mrf_normalized_error_channel(sol_2d, channel_gt)
        error_list.append(err)

    plt.figure(figsize=(7, 5))
    plt.plot(range(len(error_list)), error_list, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Error (Normalized)")
    plt.title(f"MRF Iterative Solver (sigma={sigma:.2f})")
    plt.grid(True)
    plt.show()

    # Print improvement
    if len(error_list) > 1:
        initial_err = error_list[0]
        final_err = error_list[-1]
        improvement_pct = 100.0 * (initial_err - final_err) / initial_err
        print(f"Improvement in channel: {improvement_pct:.2f}%")
    else:
        print("Only one iteration was performed; no improvement measured.")

    return channel_final, solutions_list


if __name__ == "__main__":
    run_mrf_demo()
