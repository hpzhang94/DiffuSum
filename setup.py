from setuptools import setup

setup(
    name="imp_diff",
    py_modules=["improved_diffusion"],
    install_requires=["blobfile", "torch", "tqdm"],
    #install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)
