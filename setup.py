"""Setup script for our package."""
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=["intel_aero_ros"],
    scripts=["bin/trajectory_tracking_node"],
    package_dir={"": "src"},
)

setup(**d)
