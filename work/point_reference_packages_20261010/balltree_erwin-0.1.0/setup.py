from setuptools import setup, find_packages
from Cython.Build import cythonize
from setuptools.extension import Extension
import numpy
import os
import platform

extra_compile_args = ['-O3', '-std=c++11']
extra_link_args = []

if platform.system() == 'Darwin':  # macOS
    brew_prefix = os.popen('brew --prefix').read().strip()
    libomp_path = f"{brew_prefix}/opt/libomp"
    
    extra_compile_args += ['-Xpreprocessor', '-fopenmp', f'-I{libomp_path}/include']
    extra_link_args += [f'-L{libomp_path}/lib', '-lomp']
else:
    extra_compile_args += ['-fopenmp']
    extra_link_args += ['-fopenmp']

ext_modules = [
    Extension(
        "balltree.balltree",
        ["balltree/balltree.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++"
    )
]

setup(
    name="balltree-erwin",
    version="0.1.0",
    description="Fast parallel ball tree construction for machine learning",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Maksim Zhdanov",
    author_email="maxxxzdn@pm.me",
    url="https://github.com/maxxxzdn/balltree",
    packages=find_packages(),
    ext_modules=cythonize(ext_modules),
    install_requires=[
        "numpy",
        "torch",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
    ],
    license="MIT",
)