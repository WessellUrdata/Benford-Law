from setuptools import find_packages, setup

setup(
    name="ImageForensics",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
    ],
    entry_points={
        "console_scripts": [
            # Define command-line scripts here if needed
        ],
    },
)
