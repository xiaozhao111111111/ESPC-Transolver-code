"""Build official balltree source with an LLP64-only index portability fix."""
from pathlib import Path
import shutil
import subprocess
import sys
from point_reference_runtime import ROOT, DEPS, np
from Cython.Build import cythonize
from setuptools.extension import Extension

source = ROOT / 'work/point_reference_packages_20261010/balltree_erwin-0.1.0/balltree'
build = ROOT / 'work/point_reference_packages_20261010/balltree_windows_build'
sdk = ROOT / 'work/point_reference_packages_20261010/python312_sdk/tools'
build.mkdir(exist_ok=True)
header = (source / 'balltree.h').read_text()
assert 'using IndexType = long;' in header
# Python's np.intp and Cython intp_t are 64 bit; Windows C++ long is 32 bit.
# The algorithm is unchanged. Preserve the original downloaded source.
(build / 'balltree.h').write_text(header.replace('using IndexType = long;', 'using IndexType = intptr_t;'))
shutil.copyfile(source / 'balltree.pyx', build / 'balltree.pyx')
cythonize([Extension('balltree.balltree', [str(build / 'balltree.pyx')], language='c++')], force=True)
compiler = shutil.which('clang++')
if not compiler:
    raise RuntimeError('LLVM-MinGW clang++ unavailable')
target = DEPS / 'balltree'
target.mkdir(exist_ok=True)
out = target / 'balltree.cp312-win_amd64.pyd'
command = [compiler, '-shared', '-O3', '-std=c++11', '-fno-openmp',
           '-I' + str(sdk / 'include'), '-I' + np.get_include(), '-I' + str(build),
           str(build / 'balltree.cpp'), str(sdk / 'libs/python312.lib'),
           '-L' + str(Path(compiler).parent.parent / 'lib'), '-lomp', '-o', str(out)]
print(command, flush=True)
subprocess.run(command, check=True)
shutil.copyfile(source / '__init__.py', target / '__init__.py')
# Keep portable compiler DLLs alongside the extension, without PATH/global edits.
for name in ['libomp.dll', 'libc++.dll', 'libunwind.dll']:
    dll = Path(compiler).parent / name
    if dll.exists():
        shutil.copyfile(dll, target / name)
print('BUILT', out, flush=True)
