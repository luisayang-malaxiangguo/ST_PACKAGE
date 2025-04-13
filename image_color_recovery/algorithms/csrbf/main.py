"""
CSRBF Demo Module
=================

This module provides an interactive demo for CSRBF-based image color recovery.
It follows these steps:
  1. Prompts the user to upload a color image (ground truth).
  2. Loads and preprocesses the image (converts to RGB and grayscale).
  3. Defines fixed sample indices to extract known sample color values.
  4. Runs the CSRBF pipeline (kernel generation, solving, and recovery) once.
  5. Visualizes the original image, grayscale image, and recovered color image.
"""


def run_csrbf_demo():
    """
    Run a simplified CSRBF demo that performs a one-shot recovery of
    the input image and then displays the results.
    """
    # Step 1: Upload a color image
    from google.colab import files
    print("Please upload a color image (ground truth).")
    uploaded = files.upload()
    if not uploaded:
        print("No file uploaded. Exiting demo.")
        return

    # Load and preprocess the image
    import numpy as np
    import cv2
    from skimage import color
    import matplotlib.pyplot as plt

    filename = list(uploaded.keys())[0]
    file_bytes = np.frombuffer(uploaded[filename], np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to load the uploaded image.")

    # Convert from BGR (cv2 default) to RGB and create grayscale image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_img = (color.rgb2gray(img_rgb) * 255).astype(np.uint8)

    # Define sample indices (fixed for demonstration)
    sample_indices = (np.array([10, 50, 90]), np.array([10, 50, 90]))

    # Import CSRBF functions and configuration
    from image_color_recovery.algorithms.csrbf import kernel, solver, recovery, csrbf_visualization
    from image_color_recovery.algorithms.csrbf.config import DEFAULT_PARAMS

    # Extract sample colors from the original image at given indices
    sample_colors = np.array([
        img_rgb[10, 10, :],
        img_rgb[50, 50, :],
        img_rgb[90, 90, :]
    ], dtype=np.uint8)

    # Run the CSRBF pipeline once
    # Note: All operations are executed in a one-shot manner.
    import cupy as cp
    current_samples_cp = cp.asarray(sample_colors, dtype=cp.float32)
    
    # Generate kernel matrix on the GPU
    kernel_matrix = kernel.generate_kernel_matrix_gpu(
        sample_indices,
        gray_img,
        DEFAULT_PARAMS["sigma1"],
        DEFAULT_PARAMS["sigma2"],
        DEFAULT_PARAMS["p"]
    )
    
    # Solve for interpolation coefficients on the GPU
    coeffs = solver.solve_coefficients_gpu_direct(
        kernel_matrix,
        current_samples_cp,
        DEFAULT_PARAMS["delta"]
    )
    
    # Recover the full color image using the computed coefficients
    recovered_img = recovery.recover_color_image_gpu(
        sample_indices,
        gray_img,
        coeffs,
        DEFAULT_PARAMS["sigma1"],
        DEFAULT_PARAMS["sigma2"],
        DEFAULT_PARAMS["p"],
        DEFAULT_PARAMS["batch_size"]
    )

    # Visualize the results using the CSRBF visualization module
    csrbf_visualization.visualize_results(img_rgb, gray_img, recovered_img, sample_indices)


if __name__ == "__main__":
    run_csrbf_demo()
