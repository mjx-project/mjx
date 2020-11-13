from setuptools import setup, find_packages

setup(
    name='mjconvert',
    version="0.0.3",
    description='Converter for Mahjong log files',
    author='sotetsuk',
    author_email='koyamada-s@sys.i.kyoto-u.ac.jp',
    license='MIT',
    install_requires=[],
    packages=find_packages(),
    include_package_data=True,
    package_data={'': ['.mjconvert/seed_cache/*.txt']},
    entry_points={
        'console_scripts': 'mjconvert = mjconvert.main:main'
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License"
    ]
)
