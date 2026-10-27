"""Package-relative dataset paths. Set ESPC_DATA_ROOT to use another data folder."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get('ESPC_DATA_ROOT', ROOT/'data')).resolve()
CACHE = DATA/'canonical'
SURFACE = DATA/'surface'
GEOMETRY = DATA/'geo_strong'
TESTS = {
    'id_test': CACHE/'id_test',
    'shape_ood': CACHE/'peanut_ood',
    'topology_ood': DATA/'topology/topology_ood',
    **{name: DATA/'extended'/name for name in ('n8_test', 'shape_n8_ood', 'topology_n8_ood')},
}
SPLITS = {**{s: CACHE/s for s in ('train_v3', 'train_v6s11', 'val_v3', 'val_v6s11')}, **TESTS}
