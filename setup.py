import setuptools
from torch.utils import cpp_extension

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="smyrf",
    version="0.0.1",
    author="giannisdaras",
    author_email="daras.giannhs@gmail.com",
    description="Asymmetric LSH attention.",
    long_description="Accelerate your pre-trained attention models with asymmetric LSH attention.",
    long_description_content_type="text/markdown",
    url="https://github.com/giannisdaras/smyrf",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    python_requires='>=3.6',
    ext_modules=[cpp_extension.CppExtension('smyrfsort', ['smyrfsort.cpp'])],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
