# ─────────────────────────────────────────────────────────────
# All configuration values in one place.
# ─────────────────────────────────────────────────────────────

CELL_SIZE    = 26    # pixel width/height of each grid square
DEFAULT_ROWS = 20    # starting number of rows
DEFAULT_COLS = 28    # starting number of columns
ANIM_MS      = 30    # milliseconds between animation frames (lower = faster)
SPAWN_CHANCE = 0.04  # 4% probability per agent step to spawn a new obstacle


EMPTY = 0   # passable cell
WALL  = 1   # blocked cell
START = 2   # agent start position
GOAL  = 3   # target position


# Plain flat colours — easy to read on any monitor.
COLORS = {
    EMPTY        : "#ffffff",   # white  – open cell
    WALL         : "#2d2d2d",   # dark grey – wall
    START        : "#27ae60",   # green  – start marker
    GOAL         : "#e74c3c",   # red    – goal marker
    "frontier"   : "#f39c12",   # orange – node is in the priority queue
    "visited"    : "#3498db",   # blue   – node has been expanded
    "path"       : "#2ecc71",   # bright green – final solution path
    "agent"      : "#9b59b6",   # purple – agent moving in dynamic mode
    "grid"       : "#cccccc",   # light grey – cell border lines
    "bg"         : "#f0f0f0",   # window background
    "sidebar"    : "#e8e8e8",   # sidebar panel background
}

# ─────────────────────────────────────────────────────────────
# Heuristic functions estimate the remaining distance from a
#   (A* or Greedy BFS) was chosen.
# ─────────────────────────────────────────────────────────────

import math


def manhattan(a, b):
    """
    Manhattan Distance  =  |row_a - row_b| + |col_a - col_b|

    Think of it as the number of blocks walked on a city grid
    (no diagonals).  Fast to compute and works perfectly for
    4-directional grid movement.

    Example:
        a = (0, 0),  b = (3, 4)
        manhattan(a, b) = |0-3| + |0-4| = 3 + 4 = 7
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a, b):
    """
    Euclidean Distance  =  sqrt( (row_a-row_b)^2 + (col_a-col_b)^2 )

    The straight-line geometric distance between two points.
    Slightly more accurate than Manhattan but a bit slower
    to compute due to the square root.

    Example:
        a = (0, 0),  b = (3, 4)
        euclidean(a, b) = sqrt(9 + 16) = sqrt(25) = 5.0
    """
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

