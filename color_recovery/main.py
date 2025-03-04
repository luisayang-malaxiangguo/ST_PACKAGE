import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from google.colab import files  # For Colab; modify if running locally
from tqdm import tqdm

from color_recovery.image_loader import load_image
from color_recovery.point_selector import generate_random_points
from color_recovery.kernel import generate_kernel_gpu
from color_recovery.solver import solve_coeffs_gpu
from color_recovery.reconstructor import recover_image_gpu


def main():
    """
    Run the entire color recovery pipeline.

    This function:
    1. Uploads and loads an image (in Colab).
    2. Converts the image to grayscale.
    3. Selects random points.
    4. Builds a spatial kernel matrix.
    5. Solves for coefficients using a conjugate gradient method.
    6. Reconstructs the color image and displays results.
    """
    uploaded = files.upload()
    image_path = list(uploaded.keys())[0]

    original_rgb, gray_image = load_image(image_path)

    params = {
        "percentage": 9.0,
        "sigma_1": 100.0,
        "delta": 2e-4,
        "batch_size": 10000
    }

    selected_points = generate_random_points(gray_image.shape,
                                             params["percentage"])

    print("Generating kernel matrix on GPU...")
    kernel_matrix = generate_kernel_gpu(selected_points, gray_image,
                                        params["sigma_1"])

    color_values_cpu = original_rgb[selected_points[0],
                                    selected_points[1],
                                    :].astype(np.float32)
    color_values_gpu = cp.asarray(color_values_cpu)

    print("Solving for coefficients (conjugate gradient) on GPU...")
    coeffs = solve_coeffs_gpu(kernel_matrix, color_values_gpu,
                              params["delta"])

    print("Reconstructing image on GPU...")
    recovered_image = recover_image_gpu(selected_points, gray_image,
                                        coeffs, params["sigma_1"],
                                        rescale=True,
                                        batch_size=params["batch_size"])

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 4, 1)
    plt.imshow(original_rgb)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(gray_image, cmap="gray")
    plt.title("Grayscale")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    temp_rgb = np.stack([gray_image] * 3, axis=-1)
    temp_rgb[selected_points[0], selected_points[1], :] = \
        original_rgb[selected_points[0], selected_points[1], :]
    plt.imshow(temp_rgb)
    plt.title("Random Points")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(recovered_image)
    plt.title("Recovered")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
