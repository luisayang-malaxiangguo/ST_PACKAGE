"""
Visualization functions for image color recovery in CSRBF algorithm.
"""

import matplotlib.pyplot as plt
import numpy as np


def visualize_results(original_image, gray_image, recovered_image, sample_indices):
    """
    Visualize original, grayscale, sampled, and recovered images.

    Parameters
    ----------
    original_image : np.ndarray
        Original RGB image.
    gray_image : np.ndarray
        Grayscale image.
    recovered_image : np.ndarray
        Recovered color image.
    sample_indices : tuple of np.ndarray
        Pixel indices for the sampled points (rows, cols).
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 4, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(gray_image, cmap="gray")
    plt.title("Grayscale Image")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    image_with_samples = np.stack([gray_image] * 3, axis=-1)
    image_with_samples[sample_indices[0], sample_indices[1], :] = \
        original_image[sample_indices[0], sample_indices[1], :]
    plt.imshow(image_with_samples)
    plt.title("Random Points")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(recovered_image)
    plt.title("Recovered (CSRBF)")
    plt.axis("off")

    plt.show()
