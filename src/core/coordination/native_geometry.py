"""Optional bulk C++ kernels; prediction and candidate selection stay Python.

MARCO_GEOMETRY_BACKEND=python forces the reference implementation; cpp requires
the extension; auto (default) uses it when installed, otherwise Python.
"""
import os

try:
    import _grid_geometry_native as _native
except ImportError:
    _native = None


def backend():
    mode = os.environ.get('MARCO_GEOMETRY_BACKEND', 'auto')
    if mode not in {'auto', 'python', 'cpp'}:
        raise ValueError('MARCO_GEOMETRY_BACKEND must be auto, python, or cpp')
    if mode == 'python':
        return None
    if _native is not None and _native.revision in {'grid-geometry-v3', 'grid-geometry-v4', 'grid-geometry-v5', 'grid-geometry-v6'}:
        return _native
    if mode == 'cpp':
        raise RuntimeError('Build C++ geometry: python -m pip install ./native/grid_geometry')
    return None


def backend_info():
    native = backend()
    native_astar = native is not None and os.environ.get('MARCO_NATIVE_ASTAR', '1') != '0'
    return {'backend': 'cpp' if native is not None else 'python',
            'revision': native.revision if native is not None else None,
            'native_astar': native_astar,
            'native_grid_reuse': native is not None and hasattr(native, 'prepare_grid') and
                                 os.environ.get('MARCO_NATIVE_GRID_REUSE', '1') != '0',
            'kernels': (['dijkstra-tree', 'clearance-prefixes'] +
                        (['astar-robot-bodies'] if native_astar else []) +
                        (['endpoint-margins'] if hasattr(native, 'endpoint_margins') else [])) if native is not None else []}


_PREPARED_GRIDS = {}


def prepared_grid(free_cells):
    """Process-wide immutable native adjacency for a static navigation set.

    Content-keyed and bounded; obstacles and goals are never embedded, so a
    prepared grid only removes the per-call rebuild of the free set.
    Returns ``free_cells`` itself when reuse is unavailable or disabled.
    """
    native = backend()
    if (native is None or not hasattr(native, 'prepare_grid') or
            os.environ.get('MARCO_NATIVE_GRID_REUSE', '1') == '0'):
        return free_cells
    signature = frozenset(free_cells)
    grid = _PREPARED_GRIDS.get(signature)
    if grid is None:
        if len(_PREPARED_GRIDS) >= 8:
            _PREPARED_GRIDS.clear()
        grid = _PREPARED_GRIDS[signature] = native.prepare_grid(signature)
    return grid


def prepared_astar_grid(free_cells):
    """Prepared grid for A*, which accepts it from revision v6 on."""
    native = backend()
    if native is None or native.revision != 'grid-geometry-v6':
        return free_cells
    return prepared_grid(free_cells)
