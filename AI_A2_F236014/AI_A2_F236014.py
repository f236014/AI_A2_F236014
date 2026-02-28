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


# ─────────────────────────────────────────────────────────────
# Manages the 2-D grid array and the tkinter canvas that
# draws it.  Also handles all mouse input.
# ─────────────────────────────────────────────────────────────

import random
import tkinter as tk
from constants import (
    EMPTY, WALL, START, GOAL,
    CELL_SIZE, COLORS
)


class Grid:
    """
    Holds the grid data and owns the tkinter Canvas widget.

    Attributes
    ----------
    rows, cols : grid dimensions
    grid       : 2-D list of cell codes (EMPTY / WALL / START / GOAL)
    start      : (row, col) of the start cell
    goal       : (row, col) of the goal cell
    canvas     : the tkinter Canvas this grid is drawn on
    mode_var   : tk.StringVar controlled by the draw-mode radio buttons
    """

    def __init__(self, parent_frame, rows, cols, mode_var):
        """
        Parameters
        ----------
        parent_frame : tkinter frame to place the canvas inside
        rows, cols   : initial grid size
        mode_var     : tk.StringVar — value is "wall"/"erase"/"start"/"goal"
        """
        self.rows     = rows
        self.cols     = cols
        self.mode_var = mode_var
        self.grid     = []
        self.start    = (0, 0)
        self.goal     = (rows - 1, cols - 1)

        # Create the canvas
        self.canvas = tk.Canvas(
            parent_frame,
            width             = cols * CELL_SIZE,
            height            = rows * CELL_SIZE,
            bg                = COLORS[EMPTY],
            highlightthickness = 0
        )
        self.canvas.pack(anchor="nw")

        # Bind mouse events
        self.canvas.bind("<Button-1>",  self._on_click)     # left-click
        self.canvas.bind("<B1-Motion>", self._on_drag)      # left-click drag
        self.canvas.bind("<Button-3>",  self._on_rclick)    # right-click = erase

        self.reset()

    # ── Grid initialisation ──────

    def reset(self):
        """Clear everything and place start (top-left) and goal (bottom-right)."""
        self.grid  = [[EMPTY] * self.cols for _ in range(self.rows)]
        self.start = (0, 0)
        self.goal  = (self.rows - 1, self.cols - 1)
        self.grid[0][0]                         = START
        self.grid[self.rows - 1][self.cols - 1] = GOAL
        self.redraw()

    def resize(self, rows, cols):
        """Change grid dimensions and reset."""
        self.rows = rows
        self.cols = cols
        self.canvas.config(width=cols * CELL_SIZE, height=rows * CELL_SIZE)
        self.reset()

    def random_map(self, density):
        """
        Fill grid with random walls.

        Parameters
        ----------
        density : float 0.0–1.0 — fraction of cells that become walls
        """
        self.reset()
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in (self.start, self.goal):
                    if random.random() < density:
                        self.grid[r][c] = WALL
        self.redraw()

    # ── Drawing ───────────────────────────────────────────────

    def redraw(self):
        """Redraw every cell from scratch."""
        self.canvas.delete("all")
        for r in range(self.rows):
            for c in range(self.cols):
                self.draw_cell(r, c)

    def draw_cell(self, r, c, override=None):
        """
        Draw a single cell rectangle.

        Parameters
        ----------
        r, c     : grid position
        override : optional hex colour string — lets the animator
                   colour cells without changing grid data
        """
        x1, y1 = c * CELL_SIZE, r * CELL_SIZE
        color   = override if override else COLORS[self.grid[r][c]]
        self.canvas.create_rectangle(
            x1, y1,
            x1 + CELL_SIZE, y1 + CELL_SIZE,
            fill    = color,
            outline = COLORS["grid"],
            width   = 1
        )

    def refresh_all_cells(self):
        """Redraw every cell from grid data (clears animation colours)."""
        for r in range(self.rows):
            for c in range(self.cols):
                self.draw_cell(r, c)

    # ── Mouse input ─────────────

    def _pixel_to_cell(self, event):
        """Convert a pixel (x, y) mouse position to (row, col)."""
        col = max(0, min(event.x // CELL_SIZE, self.cols - 1))
        row = max(0, min(event.y // CELL_SIZE, self.rows - 1))
        return row, col

    def _on_click(self, event):
        self._apply_draw(*self._pixel_to_cell(event))

    def _on_drag(self, event):
        self._apply_draw(*self._pixel_to_cell(event))

    def _on_rclick(self, event):
        """Right-click always erases (except start and goal cells)."""
        r, c = self._pixel_to_cell(event)
        if self.grid[r][c] not in (START, GOAL):
            self.grid[r][c] = EMPTY
            self.draw_cell(r, c)

    def _apply_draw(self, r, c):
        """
        Apply the currently selected draw mode to cell (r, c).
        mode_var is set by the Draw Mode radio buttons in the sidebar.
        """
        mode = self.mode_var.get()
        cell = self.grid[r][c]

        if mode == "wall":
            if cell in (START, GOAL): return    # never overwrite start/goal
            self.grid[r][c] = WALL
            self.draw_cell(r, c)

        elif mode == "erase":
            if cell in (START, GOAL): return
            self.grid[r][c] = EMPTY
            self.draw_cell(r, c)

        elif mode == "start":
            # remove the old start marker
            sr, sc = self.start
            self.grid[sr][sc] = EMPTY
            self.draw_cell(sr, sc)
            # place new start
            self.start        = (r, c)
            self.grid[r][c]   = START
            self.draw_cell(r, c)

        elif mode == "goal":
            # remove the old goal marker
            gr, gc = self.goal
            self.grid[gr][gc] = EMPTY
            self.draw_cell(gr, gc)
            # place new goal
            self.goal         = (r, c)
            self.grid[r][c]   = GOAL
            self.draw_cell(r, c)








