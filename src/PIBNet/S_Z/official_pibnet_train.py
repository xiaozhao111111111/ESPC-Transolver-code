"""Construct the official three-level PIBNet processor."""
import importlib.util
import sys
import types
from pathlib import Path
import torch
from PIBNet.S_Z.official_pibnet_adapter import NODE_DIM, EDGE_DIM, EDGE_TYPES
def _install_physicsnemo_activation_package_shim() -> None:
    """Expose PhysicsNeMo's exact activation factory without eager layer imports.

    PhysicsNeMo's layers package eagerly imports unrelated optional layers and
    their TorchScript kernels. PIBNet only needs ``get_activation`` here, so
    expose that unchanged factory while retaining the package path for lazy
    imports such as ``fused_silu``.
    """
    if "physicsnemo.models.layers" in sys.modules:
        return
    import physicsnemo
    import physicsnemo.models

    layer_path = Path(physicsnemo.__file__).parent / "models" / "layers"
    activation_path = layer_path / "activations.py"
    spec = importlib.util.spec_from_file_location(
        "pibnet_physicsnemo_activations", activation_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load PhysicsNeMo activation factory at {activation_path}")
    activation_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(activation_module)
    layer_package = types.ModuleType("physicsnemo.models.layers")
    layer_package.__file__ = str(layer_path / "__init__.py")
    layer_package.__path__ = [str(layer_path)]
    layer_package.__package__ = "physicsnemo.models.layers"
    layer_package.get_activation = activation_module.get_activation
    sys.modules["physicsnemo.models.layers"] = layer_package
    physicsnemo.models.layers = layer_package
_install_physicsnemo_activation_package_shim()
from gnn.pibnet import PIBNet
def make_model(device: torch.device, node_dim: int = NODE_DIM) -> PIBNet:
    return PIBNet(
        input_dim_nodes=node_dim,
        input_dim_edges={etype: EDGE_DIM for etype in EDGE_TYPES},
        output_dim=2,
        num_levels=3,
        first_top_processor_size=3,
        last_top_processor_size=3,
        bottom_processor_size=6,
        hidden_dim_scaling=2,
        distant_edge_sample_period=6,
        hidden_dim_processor=64,
        hidden_dim_node_encoder=64,
        hidden_dim_edge_encoder=64,
        hidden_dim_node_decoder=64,
        mlp_activation_fn="silu",
    ).to(device)
