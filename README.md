# Color Recovery Package

 This package performs image color recovery using GPU acceleration with CuPy. It is **recommended** to run this package in [Google Colab](https://colab.research.google.com/), where you can access a **free GPU** environment.

 It contains the following modules:

- **distance.py:** Computes GPU-based pairwise distances
- **rgb2gray.py:** Converts an RGB image to grayscale
- **image_loader.py:** Loads an image (BGR → RGB) and converts it to grayscale
- **point_selector.py:** Generates random points from the original image
- **kernel.py:** Constructs a spatial kernel matrix using a Gaussian function
- **solver.py:** Solves for coefficients using a conjugate gradient method
- **reconstructor.py:** Reconstructs the image in batches with rescaling
- **main.py:** Integrates all components to run the complete pipeline

---

## Requirements

- **Python 3.7+**

Below are the packages needed to run this project. Install with `pip` inside virtual environment.

- **OpenCV** (for image I/O):
  pip install opencv-python



- **NumPy**
  pip install numpy

- **CUDA**
  pip install cupy-cuda117

- **Matplotlib** (for visualization)
  pip install matplotlib




