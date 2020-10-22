from setuptools import setup, find_packages

setup(
    name='mjconvert',
    version="0.0.2",
    description='Converter for Mahjong log files',
    author='sotetsuk',
    author_email='koyamada-s@sys.i.kyoto-u.ac.jp',
    license='MIT',
    install_requires=[],
    packages=find_packages(),
    entry_points={
        'console_scripts': 'mjconvert = src.main:main'
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License"
    ]
)
