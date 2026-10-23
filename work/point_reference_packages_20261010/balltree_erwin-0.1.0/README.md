# Ball Tree

A fast, parallel ball tree implementation with PyTorch integration, built using Cython and OpenMP.

## Features

- **High Performance**: Optimized C++ implementation with OpenMP parallelization
- **PyTorch Integration**: Native support for PyTorch tensors
- **Flexible API**: Multiple interfaces for different use cases
- **Memory Efficient**: Efficient memory usage and management

## Installation

### From PyPI

```bash
pip install balltree-erwin
```

### From Source

```bash
git clone https://github.com/maxxxzdn/balltree.git
cd balltree
pip install -e .
```

### Requirements

- Python >= 3.8
- NumPy
- PyTorcn
- Cython >= 0.29.0 (for building from source)

### System Dependencies

#### macOS
```bash
brew install libomp
```

## Quick Start

```python
import torch
import balltree

# Create some sample data
data = torch.randn(1000, 3)  # 1000 points in 3D
batch_idx = torch.zeros(1000, dtype=torch.long)  # Single batch

# Build a ball tree
tree_indices, tree_mask = balltree.build_balltree(data, batch_idx)

# Build with rotations for cross-ball interactions
tree_idx, tree_mask, rot_indices = balltree.build_balltree_with_rotations(
    data, batch_idx, 
    strides=[2, 2], 
    ball_sizes=[64, 32, 16],
    angle=45.0
)
```

## API Reference

### Core Functions

#### `build_balltree(data, batch_idx)`
Build ball trees for batched data.

**Parameters:**
- `data`: torch.Tensor of shape (num_samples, num_features)
- `batch_idx`: torch.Tensor of shape (num_samples,) - batch assignment

**Returns:**
- `tree_indices`: Indices of tree nodes
- `tree_mask`: Boolean mask for tree structure

#### `partition_balltree(data, batch_idx, target_level)`
Partition ball trees to a specific level.

**Parameters:**
- `data`: torch.Tensor of shape (num_samples, num_features)  
- `batch_idx`: torch.Tensor of shape (num_samples,)
- `target_level`: int - target partitioning level

**Returns:**
- `partitioned_indices`: Partitioned tree indices

#### `build_balltree_with_rotations(data, batch_idx, strides, ball_sizes, angle=45.0)`
Build ball trees with rotational variants for enhanced modeling.

**Parameters:**
- `data`: torch.Tensor of shape (num_samples, num_features)
- `batch_idx`: torch.Tensor of shape (num_samples,)  
- `strides`: list of int - pooling strides
- `ball_sizes`: list of int - ball sizes for each layer
- `angle`: float - rotation angle in degrees

**Returns:**
- `tree_idx`: Original tree indices
- `tree_mask`: Tree structure mask
- `rot_tree_indices`: List of rotated tree indices

## Use Cases

This library is designed for applications requiring efficient spatial data structures:

- **Machine Learning**: Spatial attention mechanisms, geometric deep learning
- **Computer Graphics**: Collision detection, ray tracing acceleration
- **Robotics**: Path planning, nearest neighbor queries
- **Scientific Computing**: N-body simulations, clustering algorithms

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.