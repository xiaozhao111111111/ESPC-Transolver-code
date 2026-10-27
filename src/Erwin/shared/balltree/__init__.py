"""
Fast parallel ball tree construction for machine learning.

This package provides efficient ball tree construction algorithms optimized for
batch processing and parallel execution using OpenMP.
"""

__version__ = "0.1.0"

try:
    from .balltree import (
        build_balltree,
        partition_balltree,
        build_balltree_with_rotations,
        build_balltree_with_offsets,
        partition_balltree_with_offsets,
        build_balltree_with_idx,
        partition_balltree_with_idx,
        generate_rotation_matrix,
    )
except ImportError as e:
    raise ImportError(
        "Could not import compiled Cython modules. "
        "Please ensure the package was properly installed with: "
        "pip install balltree"
    ) from e

__all__ = [
    "build_balltree",
    "partition_balltree", 
    "build_balltree_with_rotations",
    "build_balltree_with_offsets",
    "partition_balltree_with_offsets",
    "build_balltree_with_idx",
    "partition_balltree_with_idx",
    "generate_rotation_matrix",
]