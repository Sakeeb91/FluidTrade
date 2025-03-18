from setuptools import setup, find_packages

setup(
    name="fluid_dynamics_hft",
    version="0.1.0",
    description="Fluid dynamics approaches to high-frequency trading",
    author="User",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "scikit-learn>=0.24.0",
    ],
    python_requires=">=3.8",
)
