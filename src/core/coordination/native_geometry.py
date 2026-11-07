"""C++ bulk geometry kernels (required); prediction and candidate selection stay in Python.

Build once with:  python -m pip install --no-build-isolation ./native/grid_geometry
"""
import os

try:
    import _grid_geometry_native as _native
except ImportError:
    _native = None

_REVISIONS = {'grid-geometry-v3', 'grid-geometry-v4', 'grid-geometry-v5'}


def backend():
    if _native is None or _native.revision not in _REVISIONS:
        raise RuntimeError('The C++ geometry module is required. Build it with: '
                           'python -m pip install --no-build-isolation ./native/grid_geometry')
    return _native


def backend_info():
    native = backend()
    native_astar = os.environ.get('MARCO_NATIVE_ASTAR', '1') != '0'
    return {'backend': 'cpp',
            'revision': native.revision,
            'native_astar': native_astar,
            'native_grid_reuse': hasattr(native, 'prepare_grid') and
                                 os.environ.get('MARCO_NATIVE_GRID_REUSE', '1') != '0',
            'kernels': (['dijkstra-tree', 'clearance-prefixes'] +
                        (['astar-robot-bodies'] if native_astar else []) +
                        (['endpoint-margins'] if hasattr(native, 'endpoint_margins') else []))}
