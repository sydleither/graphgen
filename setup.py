from setuptools import setup, Extension


def readme():
    with open('README.md') as f:
        return f.read()


# Compiler settings
extra_compile_args = ['-std=c++14', '-Wno-write-strings']
extra_link_args = ['-Wl,--verbose', '-lstdc++']  #-lc++ for clang?

# Get numpy include dirs
import numpy as np
numpy_include_path = np.get_include()

# Unweighted directed module specifications
unweighted_directed_include_dirs = [
    'graphgen/unweighted_directed',
    numpy_include_path
    ]

unweighted_directed_sources = [
    'graphgen/unweighted_directed/benchm.cpp',
    'graphgen/unweighted_directed/unweighted_directed_graph_generator.cpp'
]

unweighted_directed_module = Extension('unweighted_directed_graph_generator',
                                       language="c++14",
                                       sources=unweighted_directed_sources,
                                       extra_compile_args=extra_compile_args,
                                       include_dirs=unweighted_directed_include_dirs,
                                       extra_link_args=extra_link_args)

# Unweighted undirected module specifications
unweighted_undirected_include_dirs = [
    'graphgen/unweighted_undirected',
    numpy_include_path
]

unweighted_undirected_sources = [
    'graphgen/unweighted_undirected/benchm.cpp',
    'graphgen/unweighted_undirected/unweighted_undirected_graph_generator.cpp'
]

unweighted_undirected_module = Extension('unweighted_undirected_graph_generator',
                                       language="c++14",
                                       sources=unweighted_undirected_sources,
                                       extra_compile_args=extra_compile_args,
                                       include_dirs=unweighted_undirected_include_dirs,
                                       extra_link_args=extra_link_args)

# weighted directed module specifications
weighted_directed_include_dirs = [
    'graphgen/weighted_directed',
    numpy_include_path
]

weighted_directed_sources = [
    'graphgen/weighted_directed/benchm.cpp',
    'graphgen/weighted_directed/weighted_directed_graph_generator.cpp',
]

weighted_directed_module = Extension('weighted_directed_graph_generator',
                                         language="c++14",
                                         sources=weighted_directed_sources,
                                         extra_compile_args=extra_compile_args,
                                         include_dirs=weighted_directed_include_dirs,
                                         extra_link_args=extra_link_args)

# klemm module specifications
klemm_include_dirs = [
    'graphgen/klemm',
    numpy_include_path
]

klemm_sources = [
    'graphgen/klemm/benchm.cpp',
    'graphgen/klemm/klemm_graph_generator.cpp',
]

klemm_module = Extension('klemm_graph_generator',
                                         language="c++14",
                                         sources=klemm_sources,
                                         extra_compile_args=extra_compile_args,
                                         include_dirs=klemm_include_dirs,
                                         extra_link_args=extra_link_args)


# weighted undirected module specifications
weighted_undirected_include_dirs = [
    'graphgen/weighted_undirected',
    numpy_include_path
]

weighted_undirected_sources = [
    'graphgen/weighted_undirected/benchm.cpp',
    'graphgen/weighted_undirected/weighted_undirected_graph_generator.cpp'
]

weighted_undirected_module = Extension('weighted_undirected_graph_generator',
                                     language="c++14",
                                     sources=weighted_undirected_sources,
                                     extra_compile_args=extra_compile_args,
                                     include_dirs=weighted_undirected_include_dirs,
                                     extra_link_args=extra_link_args)

PACKAGE_NAME = 'graphgen'
setup(name='graphgen',
      version='1.0',
      description='Makes graphs with community structure',
      author='Nathaniel Rodriguez',
      packages=[PACKAGE_NAME],
      ext_package=PACKAGE_NAME,
      ext_modules=[unweighted_directed_module, unweighted_undirected_module, weighted_directed_module,
                   klemm_module, weighted_undirected_module],
      url='https://github.com/Nathaniel-Rodriguez/graphgen.git',
      install_requires=[
          'networkx',
          'numpy'
      ])
