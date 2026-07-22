from setuptools import setup, find_packages

setup(
    name="cable-threading-mvp",
    version="1.0.0",
    description="CableThreading MVP - 最小可运行线缆穿杆任务包",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "mujoco>=3.0.0",
        "h5py>=3.8.0",
        "imageio>=2.31.0",
        "imageio-ffmpeg>=0.4.0",
    ],
    extras_require={
        "lerobot": ["pyarrow>=12.0.0"],
        "ml": ["robomimic>=0.3.0", "torch>=2.0.0"],
        "all": ["pyarrow>=12.0.0", "robomimic>=0.3.0", "torch>=2.0.0"],
    },
)
