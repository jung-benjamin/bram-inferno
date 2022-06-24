from setuptools import setup, find_packages

setup(
    name = 'newbie',
    version = '0.3',
    author = 'Benjamin Jung',
    license = 'BSD-3-Clause',
    packages = find_packages(include = ['newbie', 'newbie.*']),
    description = 'A package for bayesian inference with nuclear waste.',
)
