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


# ─────────────────────────────────────────────────────────────
# Contains the two search algorithms A* and Greedy Best-First Search.
# ─────────────────────────────────────────────────────────────

import heapq                    # min-heap priority queue
from constants import WALL      # cell type code for walls


# ── Shared helper: valid neighbours of a grid cell ───────────

def get_neighbors(node, rows, cols, grid):
    """
    Yield the Up / Down / Left / Right neighbours of 'node'
    that are inside the grid AND are not walls.
    """
    r, c = node
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != WALL:
            yield (nr, nc)


# ── Shared helper: reconstruct path from came_from map ───────

def build_path(came_from, goal):
    """
    Walk came_from backwards from goal → start, then reverse.
    Returns a list of (row, col) tuples: [start, ..., goal].
    """
    path, cur = [], goal
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    return path


# ═════════════════════════════════════════════════════════════
#  ALGORITHM 1 — GREEDY BEST-FIRST SEARCH
#    Priority = h(n)  
# ═════════════════════════════════════════════════════════════

def greedy_bfs(grid, start, goal, h_fn):
    """
    Parameters
    ----------
    grid   : 2-D list of cell codes (EMPTY / WALL / START / GOAL)
    start  : (row, col) tuple
    goal   : (row, col) tuple
    h_fn   : heuristic function  e.g. manhattan or euclidean

    Returns
    -------
    path           : list of (row,col) from start to goal, or None
    visited_order  : order in which nodes were expanded (for animation)
    frontier_snaps : snapshots of frontier at each step  (for animation)
    """
    rows, cols = len(grid), len(grid[0])

    # Each entry in the priority queue: (h_value, node)
    pq = [(h_fn(start, goal), start)]

    came_from = {start: None}   # maps node → parent node
    visited   = {start}         # strict visited set

    visited_order  = []         # for animation
    frontier_snaps = []         # for animation

    while pq:
        # ── Pop the node with the LOWEST h(n) ────────────────
        _, cur = heapq.heappop(pq)
        visited_order.append(cur)

        if cur == goal:
            return build_path(came_from, goal), visited_order, frontier_snaps

        for nb in get_neighbors(cur, rows, cols, grid):
            if nb not in visited:
                visited.add(nb)
                came_from[nb] = cur
                heapq.heappush(pq, (h_fn(nb, goal), nb))

        # Save a frontier snapshot for the animator
        frontier_snaps.append([x[1] for x in pq])

    return None, visited_order, frontier_snaps   # no path exists


# ═════════════════════════════════════════════════════════════
#  ALGORITHM 2 — A* SEARCH
#    f(n) = g(n) + h(n)
# ═════════════════════════════════════════════════════════════

def astar(grid, start, goal, h_fn):
    """
    Parameters
    ----------
    grid   : 2-D list of cell codes
    start  : (row, col) tuple
    goal   : (row, col) tuple
    h_fn   : heuristic function  e.g. manhattan or euclidean

    Returns
    -------
    path           : optimal list of (row,col), or None
    visited_order  : expansion order  (for animation)
    frontier_snaps : frontier snapshots (for animation)
    """
    rows, cols = len(grid), len(grid[0])

    g = {start: 0}              # best known g-cost to each node

    # Each entry: (f_value, g_value, node)
    # g_value is stored as a tiebreaker — prefer nodes with higher g
   
    pq = [(h_fn(start, goal), 0, start)]

    came_from = {start: None}
    expanded  = set()           # nodes fully processed

    visited_order  = []
    frontier_snaps = []

    while pq:
        # ── Pop the node with the LOWEST f = g + h ───────────
        _, g_cur, cur = heapq.heappop(pq)

        if cur in expanded:
            # A cheaper path to this node was already processed — skip
            continue

        expanded.add(cur)
        visited_order.append(cur)

        if cur == goal:
            return build_path(came_from, goal), visited_order, frontier_snaps

        for nb in get_neighbors(cur, rows, cols, grid):
            new_g = g[cur] + 1          # uniform step cost = 1

            # Only update if we found a cheaper route to this neighbour
            if nb not in g or new_g < g[nb]:
                g[nb]         = new_g
                came_from[nb] = cur
                f             = new_g + h_fn(nb, goal)
                heapq.heappush(pq, (f, new_g, nb))

        frontier_snaps.append([x[2] for x in pq])

    return None, visited_order, frontier_snaps   # no path exists




