from setuptools import setup, find_packages
import io

with io.open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sdsofa",
    version="0.0.1",
    author="Joshua Fishman, Samuel Ubellacker",
    author_email="subella@mit.edu",
    description="Simulation code for the Soft Drone in SOFA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=2",
    install_requires=["cvxopt==1.2.6",
                      "importlib-resources==3.3.1",
                      "meshio==2.3.10",
                      "numpy==1.16.6",
                      "scipy==1.2.3"
                     ],
)
