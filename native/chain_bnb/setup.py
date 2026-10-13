"""Optional native backend; install with the simulator's Python interpreter."""
import os
from pathlib import Path
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

os.chdir(Path(__file__).resolve().parent)
setup(
    name="ai2thor-chain-bnb-native",
    version="0.2.0",
    ext_modules=[Pybind11Extension(
        "_chain_bnb_native", ["chain_bnb.cpp"], cxx_std=17,
        extra_compile_args=["-O3", "-fno-fast-math", "-ffp-contract=off"]
        if os.name != "nt" else ["/O2", "/fp:strict"],
    )],
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.10",
)
