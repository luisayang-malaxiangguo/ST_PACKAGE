# image_color_recovery/algorithms/csrbf/main.py

def run_csrbf_demo():
    """
    Runs an interactive demo for the CSRBF-based image color recovery.
    This function uploads an image, processes it, computes the CSRBF
    interpolation, recovers the color image, and visualizes the results.
    """
    # Step 1: Upload an image from your local machine
    from google.colab import files
    print("Please upload a color image (this will be used as the ground truth).")
    uploaded = files.upload()
    
    if not uploaded:
        print("No file uploaded. Exiting demo.")
        return
    
    # Step 2: Load and preprocess the image
    import numpy as np
    import cv2
    from skimage import color
    import cupy as cp

    # Assume a single uploaded file; get its filename
    filename = list(uploaded.keys())[0]
    file_bytes = np.frombuffer(uploaded[filename], np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to read the uploaded image.")
    
    # Convert BGR (cv2 default) to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale; scale to 0-255
    gray_img = (color.rgb2gray(img_rgb) * 255).astype(np.uint8)

    # Step 3: Define sample indices (for demo purposes, fixed indices are used)
    # You may modify these indices or use a random sampling function if needed.
    sample_indices = (np.array([10, 50, 90]), np.array([10, 50, 90]))

    # Step 4: Import CSRBF modules and configuration
    from image_color_recovery.algorithms.csrbf import kernel, solver, recovery, csrbf_visualization
    from image_color_recovery.algorithms.csrbf.config import DEFAULT_PARAMS

    # Generate the CSRBF kernel matrix on the GPU
    kernel_matrix = kernel.generate_kernel_matrix_gpu(
        sample_indices,
        gray_img,
        DEFAULT_PARAMS["sigma1"],
        DEFAULT_PARAMS["sigma2"],
        DEFAULT_PARAMS["p"]
    )

    # Step 5: Extract sample color values from the original image using the chosen indices
    sampled_colors = np.array([
        img_rgb[10, 10, :],
        img_rgb[50, 50, :],
        img_rgb[90, 90, :]
    ], dtype=np.uint8)
    sampled_colors_cp = cp.asarray(sampled_colors, dtype=cp.float32)

    # Solve for the interpolation coefficients using the kernel matrix and sample colors
    coeffs = solver.solve_coefficients_gpu_direct(
        kernel_matrix,
        sampled_colors_cp,
        DEFAULT_PARAMS["delta"]
    )

    # Step 6: Recover the full color image by applying the interpolation
    recovered_img = recovery.recover_color_image_gpu(
        sample_indices,
        gray_img,
        coeffs,
        DEFAULT_PARAMS["sigma1"],
        DEFAULT_PARAMS["sigma2"],
        DEFAULT_PARAMS["p"],
        DEFAULT_PARAMS["batch_size"]
    )

    # Step 7: Visualize results using the csrbf visualization module
    csrbf_visualization.visualize_results(img_rgb, gray_img, recovered_img, sample_indices)

# Allow running the demo as a standalone script:
if __name__ == "__main__":
    run_csrbf_demo()
