from setuptools import setup, find_packages

setup(
    name="jax-noprop",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jax>=0.4.0",
        "jaxlib>=0.4.0",
        "flax>=0.7.0",
        "numpy>=1.20.0",
    ],
)
