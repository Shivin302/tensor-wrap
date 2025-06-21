from setuptools import setup, find_packages

setup(
    name="tensorwrap",
    version="0.1.0",
    packages=find_packages(include=["tensorwrap", "tensorwrap.*"]),
    py_modules=["tensorwrap", "interactive_tensorwrap"],
    entry_points={
        "console_scripts": [
            "tensorwrap=tensorwrap.cli:main",
        ],
    },
    install_requires=[
        "tqdm",
    ],
    python_requires=">=3.6",
)
