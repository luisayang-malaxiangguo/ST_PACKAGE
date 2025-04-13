# setup.py
from setuptools import setup, find_packages

setup(
    name='color_recovery',
    version='0.1.0',
    packages=find_packages(exclude=['tests*']),
    install_requires=[
        'opencv-python',
        'scikit-image',
        'numpy',
        'matplotlib',
        'tqdm',
        'scipy>=1.8.0'
        # Typically you do NOT include google-colab in install_requires
        # because Colab already provides it.
    ],
    extras_require={
        "colab": ["google-colab"],
        # CUDA 10.x series
        'cuda100': ['cupy-cuda100'],  # CUDA 10.0
        'cuda101': ['cupy-cuda101'],  # CUDA 10.1
        'cuda102': ['cupy-cuda102'],  # CUDA 10.2

        # CUDA 11.x series (listed by minor version)
        'cuda110': ['cupy-cuda110'],  # CUDA 11.0
        'cuda111': ['cupy-cuda111'],  # CUDA 11.1
        'cuda112': ['cupy-cuda112'],  # CUDA 11.2
        'cuda113': ['cupy-cuda113'],  # CUDA 11.3
        'cuda114': ['cupy-cuda114'],  # CUDA 11.4
        'cuda115': ['cupy-cuda115'],  # CUDA 11.5
        'cuda116': ['cupy-cuda116'],  # CUDA 11.6
        'cuda117': ['cupy-cuda117'],  # CUDA 11.7
        'cuda118': ['cupy-cuda118'],  # CUDA 11.8

        # CUDA 12.x series (listed by minor version)
        'cuda120': ['cupy-cuda120'],  # CUDA 12.0
        'cuda121': ['cupy-cuda121'],  # CUDA 12.1
        'cuda122': ['cupy-cuda122'],  # CUDA 12.2
        'cuda123': ['cupy-cuda123'],  # CUDA 12.3
        'cuda124': ['cupy-cuda124'],  # CUDA 12.4
        'cuda125': ['cupy-cuda125'],  # CUDA 12.5
        'cuda126': ['cupy-cuda126'],  # CUDA 12.6
        'cuda127': ['cupy-cuda127'],  # CUDA 12.7
        'cuda128': ['cupy-cuda128'],  # CUDA 12.8
        'cuda129': ['cupy-cuda129'],  # CUDA 12.9
        'cuda130': ['cupy-cuda130'],  # CUDA 13.0 (future or specialized builds)
    }
)
