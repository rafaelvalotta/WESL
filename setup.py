from setuptools import setup, find_packages

setup(
    name='WESL',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'numpy',
        'xarray',
        'matplotlib',
        'scipy',
        'pyproj',
        'openmdao',
        'pywake'
    ],
    author='Rafael Valotta Rodrigues',
    author_email='r.valottarodrigues@umb.edu',
    description='Optimizer for offshore systems.',
    url='https://github.com/rafaelvalotta/WESL',
    classifiers=[
        'Programming Language :: Python :: 3.11.11',
    ],
)
