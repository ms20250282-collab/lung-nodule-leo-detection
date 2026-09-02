from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="lung-nodule-leo-detection",
    version="1.0.0",
    author="Medical AI Research Team",
    description="Enhanced Deep Feature Selection Using LEO for Lung Nodule Detection in CT Images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ms20250282-collab/lung-nodule-leo-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical/Dental",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "leo-detect=scripts.train_detector:main",
            "leo-classify=scripts.train_classifier:main",
            "leo-select=scripts.apply_leo_selection:main",
            "leo-evaluate=scripts.evaluate_model:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
