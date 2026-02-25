"""Grid pattern transformation tasks (ARC-like).

The model receives a transformation code and an input grid, and must
produce the transformed output grid. Cells use digits 0-9 as colors.

Format: h01\n10=10\n01  (h = horizontal flip, grid rows separated by newlines)

Transformation codes:
    c = copy (identity)
    h = horizontal flip
    v = vertical flip
    r = rotate 90 degrees clockwise
    s = rotate 180 degrees
    q = rotate 270 degrees clockwise
"""

import numpy as np

_TRANSFORMS: dict[str, str] = {
    "c": "identity",
    "h": "hflip",
    "v": "vflip",
    "r": "rot90",
    "s": "rot180",
    "q": "rot270",
}


def grid_transform(rng: np.random.Generator, level: int) -> str:
    """Grid transformation. Level = grid size (2-8)."""
    size = max(2, level)
    n_colors = min(3 + level // 2, 10)
    grid = rng.integers(0, n_colors, size=(size, size))

    code = list(_TRANSFORMS.keys())[int(rng.integers(len(_TRANSFORMS)))]
    transform = _TRANSFORMS[code]

    if transform == "identity":
        out = grid.copy()
    elif transform == "hflip":
        out = np.fliplr(grid)
    elif transform == "vflip":
        out = np.flipud(grid)
    elif transform == "rot90":
        out = np.rot90(grid, k=-1)
    elif transform == "rot180":
        out = np.rot90(grid, k=2)
    elif transform == "rot270":
        out = np.rot90(grid, k=1)
    else:
        raise ValueError(f"Unknown transform: {transform}")

    return f"{code}{_grid_to_str(grid)}={_grid_to_str(out)}"


def _grid_to_str(grid: np.ndarray) -> str:
    """Render grid as rows of digits separated by newlines."""
    return "\n".join("".join(str(int(c)) for c in row) for row in grid)
