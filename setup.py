from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name="cctc2",
      ext_modules=[cpp_extension.CppExtension(
          "cctc2",
          ["cctc2.cpp"],
          extra_compile_args=["-g"],
      )],
      cmdclass={"build_ext": cpp_extension.BuildExtension})
