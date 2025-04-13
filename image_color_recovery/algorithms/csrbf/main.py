"""
CSRBF Demo Module
=================

This module provides an interactive demo for CSRBF-based image color recovery.
The demo performs an iterative refinement of the recovered image by updating a
set of sample colors, computes a GPU-based error at each iteration, and then
plots the error versus iteration. Finally, it visualizes the original image,
the grayscale image, and the recovered image.

The code is intended to be run in an interactive environment such as Google Colab.
"""


def compute_error_gpu(ground_truth, recovered_image, image_shape):
    """
    Compute the GPU-based Frobenius norm error between the ground truth and
    recovered image.

    Parameters
    ----------
    ground_truth : np.ndarray
        The original image (ground truth) as a NumPy array.
    recovered_image : np.ndarray
        The reconstructed image as a NumPy array.
    image_shape : tuple of int
        The (rows, cols) shape of the image.

    Returns
    -------
    float
        The normalized error computed on the CPU.
    """
    import cupy as cp

    # Total number of pixels
    m = image_shape[0] * image_shape[1]

    # Convert images to CuPy arrays and cast to float32 for efficiency
    gt_cp = cp.asarray(ground_truth, dtype=cp.float32)
    rec_cp = cp.asarray(recovered_image, dtype=cp.float32)

    # Reshape both arrays to column vectors
    gt_cp = gt_cp.reshape(-1, 1)
    rec_cp = rec_cp.reshape(-1, 1)

    # Compute the Frobenius norm of the difference, normalized by (3 * m)
    diff_cp = gt_cp - rec_cp
    norm_cp = cp.linalg.norm(diff_cp, ord='fro') / (3.0 * m)

    # Return the computed error to CPU
    return float(norm_cp.get())


def run_csrbf_demo():
    """
    Run an interactive CSRBF demo.

    The demo performs the following steps:
      1. Prompts the user to upload a color image (ground truth).
      2. Loads and preprocesses the image, converting it to RGB and grayscale.
      3. Defines sample indices to extract known color values.
      4. Iteratively runs the CSRBF recovery pipeline while updating sample
         colors and recording the error between the recovered image and the
         original image.
      5. Plots the error (Frobenius norm) versus the iteration number.
      6. Visualizes the final result, including the original, grayscale, and
         recovered images.
    """
    # Step 1: Upload a color image
    from google.colab import files
    print("Please upload a color image (as ground truth).")
    uploaded = files.upload()
    if not uploaded:
        print("No file uploaded. Exiting demo.")
        return

    # Step 2: Load and preprocess the image
    import numpy as np
    import cv2
    from skimage import color
    import cupy as cp
    import matplotlib.pyplot as plt

    filename = list(uploaded.keys())[0]
    file_bytes = np.frombuffer(uploaded[filename], dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to load the uploaded image.")

    # Convert the image from BGR (default in cv2) to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Generate a grayscale version scaled to 0-255
    gray_img = (color.rgb2gray(img_rgb) * 255).astype(np.uint8)
    image_shape = gray_img.shape  # (rows, cols)

    # Step 3: Define sample indices (using fixed indices for demonstration)
    sample_indices = (np.array([10, 50, 90]), np.array([10, 50, 90]))

    # Step 4: Import CSRBF functions and configuration
    from image_color_recovery.algorithms.csrbf import kernel, solver, recovery, csrbf_visualization
    from image_color_recovery.algorithms.csrbf.config import DEFAULT_PARAMS

    # Initialize the known sample colors using the original image
    original_samples = np.array([
        img_rgb[10, 10, :],
        img_rgb[50, 50, :],
        img_rgb[90, 90, :]
    ], dtype=np.uint8)
    current_samples = original_samples.copy()

    # Step 5: Iterative refinement loop with error tracking
    error_list = []
    iterations = 5  # Adjust the number of iterations as needed
    for i in range(iterations):
        print(f"Iteration {i + 1}")

        # Convert current sample colors to a Cupy array (float32)
        current_samples_cp = cp.asarray(current_samples, dtype=cp.float32)

        # Generate the kernel matrix using current configuration
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

        # Compute the error between the recovered image and the ground truth
        current_error = compute_error_gpu(img_rgb, recovered_img, image_shape)
        error_list.append(current_error)
        print(f"Iteration {i + 1} - Error: {current_error:.4f}")

        # Update the sample colors with the recovered image values at the sample indices
        current_samples = np.array([
            recovered_img[10, 10, :],
            recovered_img[50, 50, :],
            recovered_img[90, 90, :]
        ], dtype=np.uint8)

    # Step 6: Plot error vs. iteration
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, iterations + 1), error_list, marker='o', linestyle='-')
    plt.xlabel("Iteration")
    plt.ylabel("Error (Frobenius norm)")
    plt.title("CSRBF Demo: Error vs. Iteration")
    plt.grid(True)
    plt.show()

    # Step 7: Final visualization of the recovered results
    csrbf_visualization.visualize_results(img_rgb, gray_img, recovered_img, sample_indices)


if __name__ == "__main__":
    run_csrbf_demo()
