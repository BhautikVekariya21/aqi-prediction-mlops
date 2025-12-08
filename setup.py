"""
Setup configuration for aqi-prediction-mlops package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="aqi-prediction-mlops",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-grade AQI prediction system for India with MLOps pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/aqi-prediction-mlops",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9,<3.12",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "aqi-train=src.pipeline.stage_06_model_training:main",
            "aqi-predict=src.api.main:main",
        ],
    },
)