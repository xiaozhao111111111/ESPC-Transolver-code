# Ball-tree binding

Erwin uses the author's ball-tree partition and rotation/coarsening functions.
The source is included in `balltree/`. The C ABI avoids depending on a particular
CPython/Cython extension build. Its index type is pointer-sized on Windows.

The full matching Windows runtime includes the compiled library. For a
source-only installation, install a C++ compiler and run from the package root:

```sh
python scripts/train_erwin.py --build-native
```

Supported compiler entry points are `clang++`, `g++`, or `cl` in an MSVC developer
shell. The library is written under `runtime/native/balltree_build/`; source
files are not modified. Set `ESPC_BALLTREE_LIBRARY` to select a library at another
location. Compiler runtime libraries must also be available to the loader.

The provided data contain cached trees for Geo-strong/surface inputs; S(Z)
inputs build their scene trees during loading. All branches retain the same
partition algorithm and original point ordering/masks.
