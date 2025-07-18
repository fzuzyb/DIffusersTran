#!/usr/bin/env python

from setuptools import find_packages, setup

import os
import subprocess
import time

version_file = 'version.py'


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def get_git_hash():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        sha = out.strip().decode('ascii')
    except OSError:
        sha = 'unknown'

    return sha


def get_hash():
    if os.path.exists('.git'):
        sha = get_git_hash()[:7]
    # currently ignore this
    # elif os.path.exists(version_file):
    #     try:
    #         from core.version import __version__
    #         sha = __version__.split('+')[-1]
    #     except ImportError:
    #         raise ImportError('Unable to get git version')
    else:
        sha = 'unknown'

    return sha


def write_version_py():
    content = """# GENERATED VERSION FILE
# TIME: {}
__version__ = '{}'
__gitsha__ = '{}'
version_info = ({})
"""
    sha = get_hash()
    with open('VERSION', 'r') as f:
        SHORT_VERSION = f.read().strip()
    VERSION_INFO = ', '.join(
        [x if x.isdigit() else f'"{x}"' for x in SHORT_VERSION.split('.')]
    )

    version_file_str = content.format(time.asctime(), SHORT_VERSION, sha, VERSION_INFO)
    with open(version_file, 'w') as f:
        f.write(version_file_str)


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def make_cuda_ext(name, module, sources, sources_cuda=None):
    if sources_cuda is None:
        sources_cuda = []
    define_macros = []
    extra_compile_args = {'cxx': []}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension

    return extension(
        name=f'{module}.{name}',
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


def get_requirements(filename='requirements.txt'):
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, filename), 'r') as f:
        requires = [line.replace('\n', '') for line in f.readlines()]
    return requires


if __name__ == '__main__':
    # cuda_ext = os.getenv('IVCORE_EXT')  # whether compile cuda ext
    cuda_ext = 'False'
    if cuda_ext == 'True':
        try:
            import torch
            from torch.utils.cpp_extension import (
                BuildExtension,
                CppExtension,
                CUDAExtension,
            )
        except ImportError:
            raise ImportError(
                'Unable to import torch - torch is needed to build cuda extensions'
            )

        ext_modules = [
            make_cuda_ext(
                name='deform_conv_ext',
                module='core.ops.dcn',
                sources=['src/deform_conv_ext.cpp'],
                sources_cuda=[
                    'src/deform_conv_cuda.cpp',
                    'src/deform_conv_cuda_kernel.cu',
                ],
            ),
            make_cuda_ext(
                name='fused_act_ext',
                module='core.ops.fused_act',
                sources=['src/fused_bias_act.cpp'],
                sources_cuda=['src/fused_bias_act_kernel.cu'],
            ),
            make_cuda_ext(
                name='upfirdn2d_ext',
                module='core.ops.upfirdn2d',
                sources=['src/upfirdn2d.cpp'],
                sources_cuda=['src/upfirdn2d_kernel.cu'],
            ),
        ]
        setup_kwargs = dict(cmdclass={'build_ext': BuildExtension})
    else:
        ext_modules = []
        setup_kwargs = dict()

    #write_version_py()
    setup(
        name='ivface',
        version=get_version(),
        description='Imperial Vision Technology AI Algorithm Hub',
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='Wei Deng and the Algorithm Team',
        author_email='weideng.chn@gmail.com',
        keywords='computer vision',
        url='https://www.imperial-vision.com/',
        include_package_data=True,
        packages=find_packages(),
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: Other/Proprietary License',
            'Environment :: GPU :: NVIDIA CUDA',
            'Framework :: Flake8',
            'Operating System :: OS Independent',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.10',
        ],
        license='Proprietary',
        setup_requires=['cython', 'numpy', 'torch'],
        # install_requires=get_requirements(),
        ext_modules=ext_modules,
        zip_safe=False,
        **setup_kwargs,
    )
