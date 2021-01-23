import setuptools


with open("VERSION", "r", encoding="utf-8") as fv:
    version = fv.read()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cv-models",
    version="0.0.1",
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
