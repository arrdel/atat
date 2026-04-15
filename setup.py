"""
Setup script for ATAT: Adaptive Token Attention for Text Diffusion.

Implementation of "Not All Tokens Are Equal: Importance-Aware Masking
for Discrete Diffusion Language Models" (NeurIPS 2026).
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#") and not line.startswith("-")
    ]

setup(
    name="atat-diffusion",
    version="1.0.0",
    author="Adele Chinda",
    description="Not All Tokens Are Equal: Importance-Aware Masking for Discrete Diffusion Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arrdel/atat",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    keywords="diffusion language models, discrete diffusion, text generation, adaptive masking",
)
