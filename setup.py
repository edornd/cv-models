import os
import setuptools


PACKAGE = "cvmodels"
VER_FILE = "version.py"


def read_version(package: str, version_file: str) -> str:
    version_str = "unknown"
    version_path = os.path.join(package, version_file)
    try:
        version_line = open(version_path, "rt").read()
        version_str = version_line.split("=")[-1].rstrip().replace('"', "")
        return version_str
    except EnvironmentError:
        raise RuntimeError(f"Unable to find {version_path} or file is not well formed.")


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setuptools.setup(
    name="cv-models",
    version=read_version(PACKAGE, VER_FILE),
    author="Edoardo Arnaudo",
    author_email="edoardo.arnaudo@polito.it",
    description="Implementations of common Computer Vision models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/edornd/cv-models",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch",
        "torchvision",
    ],
    python_requires='>=3.6',)
