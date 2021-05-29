from setuptools import find_packages, setup

setup(
    name="mjx",
    version="0.0.1",
    description="mjx",
    author="Sotetsu KOYAMADA",
    author_email="koyamada-s@sys.i.kyoto-u.ac.jp",
    license="MIT",
    install_requires=[],
    packages=find_packages(),
    include_package_data=True,
    entry_points={"console_scripts": "mjx = mjx.main:cli"},
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
    ],
)
