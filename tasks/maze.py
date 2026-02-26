"""Maze pathfinding tasks.

Generates random perfect mazes via randomized DFS and finds the shortest
path from start (s) to end (e) using BFS. The answer is a sequence of
room-to-room directions (u/d/l/r).

Format:
    #####
    #s.e#
    #.#.#
    #...#
    #####
    =rrd

Grid uses: # (wall), . (open), s (start), e (end)
The maze interior is level x level rooms in a (2*level+1) x (2*level+1) grid.
"""

from collections import deque

import numpy as np


def maze_path(rng: np.random.Generator, level: int) -> str:
    """Maze pathfinding. Level = interior size (2-10)."""
    size = max(2, min(level, 10))
    maze = _generate_maze(rng, size, size)

    # Start = top-left room, End = bottom-right room
    start_room = (0, 0)
    end_room = (size - 1, size - 1)
    maze[1, 1] = "s"
    maze[2 * size - 1, 2 * size - 1] = "e"

    path = _bfs_rooms(maze, size, size, start_room, end_room)

    maze_str = "\n".join("".join(row) for row in maze)
    dir_str = "".join(path) if path else "n"
    return f"{maze_str}={dir_str}"


def _generate_maze(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """Generate a perfect maze using randomized DFS.

    Returns a (2h+1) x (2w+1) character grid. Room cells are at odd indices,
    wall cells at even indices.
    """
    rows, cols = 2 * h + 1, 2 * w + 1
    maze = np.full((rows, cols), "#", dtype="U1")

    # Carve rooms and passages via DFS
    visited = np.zeros((h, w), dtype=bool)
    stack = [(0, 0)]
    visited[0, 0] = True
    maze[1, 1] = "."

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while stack:
        cy, cx = stack[-1]
        neighbors = []
        for dy, dx in directions:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                neighbors.append((ny, nx))

        if neighbors:
            ny, nx = neighbors[int(rng.integers(len(neighbors)))]
            # Remove wall between current and neighbor
            maze[2 * cy + 1 + (ny - cy), 2 * cx + 1 + (nx - cx)] = "."
            maze[2 * ny + 1, 2 * nx + 1] = "."
            visited[ny, nx] = True
            stack.append((ny, nx))
        else:
            stack.pop()

    return maze


def _bfs_rooms(
    maze: np.ndarray,
    h: int,
    w: int,
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[str] | None:
    """BFS at room level. Returns list of directions (u/d/l/r) or None."""
    visited = set()
    visited.add(start)
    queue: deque[tuple[tuple[int, int], list[str]]] = deque([(start, [])])

    moves = {"u": (-1, 0), "d": (1, 0), "l": (0, -1), "r": (0, 1)}

    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == end:
            return path
        for direction, (dr, dc) in moves.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                # Check wall between rooms in the grid
                wall_r = 2 * r + 1 + dr
                wall_c = 2 * c + 1 + dc
                if maze[wall_r, wall_c] != "#":
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [direction]))

    return None
