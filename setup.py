# -*- coding: utf-8 -*-

# Copied from github.com/pybind/cmake_example
# Copyright (c) 2016 The Pybind Development Team, All rights reserved.
# https://github.com/pybind/cmake_example/blob/master/LICENSE
import os
import re
import subprocess
import sys

from pathlib import Path
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def initialize_options(self):
        build_ext.initialize_options(self)
        self.use_system_boost = None
        self.use_system_grpc = None

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cfg = "Debug" if self.debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # from Python.
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(extdir),
            "-DPYTHON_EXECUTABLE={}".format(sys.executable),
            # not used on MSVC, but no harm
            "-DCMAKE_BUILD_TYPE={}".format(cfg),
            "-DMJX_BUILD_TESTS=OFF",
            "-DMJX_BUILD_PYTHON=ON",
        ]
        build_args = []

        build_boost = "ON"
        if "MJX_BUILD_BOOST" in os.environ:
            assert os.environ["MJX_BUILD_BOOST"] in ["OFF", "ON"]
            build_boost = os.environ["MJX_BUILD_BOOST"]
        cmake_args.append(f"-DMJX_BUILD_BOOST={build_boost}")

        build_grpc = "ON"
        if "MJX_BUILD_GRPC" in os.environ:
            assert os.environ["MJX_BUILD_GRPC"] in ["OFF", "ON"]
            build_grpc = os.environ["MJX_BUILD_GRPC"]
        cmake_args.append(f"-DMJX_BUILD_GRPC={build_grpc}")

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator:
                try:
                    import ninja  # noqa: F401

                    cmake_args += ["-GNinja"]
                except ImportError:
                    pass

        else:

            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
                ]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += ["-j{}".format(self.parallel)]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="mjx",
    version="0.1.0",
    author="Mjx Project Team",
    author_email="koyamada-s@sys.i.kyoto-u.ac.jp",
    description="",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages("."),
    package_dir={"": "."},
    package_data={"mjx": ["visualizer/*.svg", "visualizer/GL-MahjongTile.ttf"]},
    # package_data={'': ['*.json']},
    cmdclass={"build_ext": CMakeBuild},
    # TODO: remove MJX_DIR (by removing cache?)
    ext_modules=[CMakeExtension("_mjx")],
    zip_safe=False,
    install_requires=[
        "protobuf==3.20.2",
        "google",
        "grpcio",
        "numpy",
        "pillow",
        "svgwrite",
        "inquirer",
    ],
    extras_require={"test": ["pytest"]},
    include_package_data=True,
)
