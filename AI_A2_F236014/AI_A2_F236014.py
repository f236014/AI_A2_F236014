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

    
# ─────────────────────────────────────────────────────────────
# Handles two phases of animation
#
#   Phase 1 — _animate_search()
#   Phase 2 — _move_agent()  
#
# Uses tkinter's root.after() for non-blocking animation
# so the GUI stays responsive during the search.
# ─────────────────────────────────────────────────────────────

import random
from tkinter import messagebox
from constants import (
    EMPTY, WALL, START, GOAL,
    ANIM_MS, SPAWN_CHANCE, COLORS
)


class Animator:
    """
    Drives the step-by-step search animation and dynamic agent movement.

    Parameters
    ----------
    root       : tkinter root window (needed for root.after())
    grid_obj   : Grid instance (owns grid data + canvas drawing)
    algo_var   : tk.StringVar — "A*" or "Greedy BFS"
    h_fn_getter: callable that returns the current heuristic function
    metrics    : dict with keys "nodes", "cost", "time" (tk.StringVar)
    """

    def __init__(self, root, grid_obj, algo_var, h_fn_getter, metrics):
        self.root        = root
        self.grid_obj    = grid_obj
        self.algo_var    = algo_var
        self.h_fn_getter = h_fn_getter
        self.metrics     = metrics

        self.running   = False   # True while animation is active
        self.anim_id   = None    # ID from root.after() — lets us cancel

        # Agent state (used only in dynamic mode)
        self.agent_pos = None
        self.cur_path  = []
        self.path_idx  = 0

    # ── Public control methods ────────────────────────────────

    def stop(self):
        """Cancel any running animation immediately."""
        self.running = False
        if self.anim_id:
            self.root.after_cancel(self.anim_id)
            self.anim_id = None

    def run(self, path, visited_order, frontier_snaps, dynamic_mode):
        """
        Start the animation for a completed search.

        Parameters
        ----------
        path           : solution path (list of (row,col)), or None
        visited_order  : nodes in the order they were expanded
        frontier_snaps : frontier state at each expansion step
        dynamic_mode   : bool — if True, move the agent after drawing path
        """
        self.running = True

        # Store path so the agent can follow it in dynamic mode
        self.cur_path  = path if path else []
        self.agent_pos = self.grid_obj.start
        self.path_idx  = 0

        self._animate_search(visited_order, frontier_snaps, path,
                              dynamic_mode, step=0)

    # ── Phase 1: animate the search expansion ─────────────────

    def _animate_search(self, visited_order, frontier_snaps, path,
                        dynamic_mode, step):
        """
        Reveal nodes one at a time.
        Uses root.after() so the window stays responsive.
        """
        if not self.running:
            return

        if step < len(visited_order):
            # Colour the newly expanded node blue
            r, c = visited_order[step]
            if self.grid_obj.grid[r][c] not in (START, GOAL):
                self.grid_obj.draw_cell(r, c, COLORS["visited"])

            # Colour frontier (queued) nodes orange
            if step < len(frontier_snaps):
                for fr, fc in frontier_snaps[step]:
                    if self.grid_obj.grid[fr][fc] not in (START, GOAL, WALL):
                        self.grid_obj.draw_cell(fr, fc, COLORS["frontier"])

            # Schedule next frame
            self.anim_id = self.root.after(
                ANIM_MS,
                lambda: self._animate_search(
                    visited_order, frontier_snaps, path, dynamic_mode, step + 1)
            )

        else:
            # ── Draw the solution path in green
            if path:
                for r, c in path:
                    if self.grid_obj.grid[r][c] not in (START, GOAL):
                        self.grid_obj.draw_cell(r, c, COLORS["path"])

            if dynamic_mode and path:
                # Begin agent movement
                self.path_idx = 0
                self.anim_id  = self.root.after(ANIM_MS * 3, self._move_agent)
            else:
                self.running = False

    # ── Phase 2: move the agent (dynamic mode only)

    def _move_agent(self):
        """
        Advance the agent one cell along cur_path.
        After each step:
          - Try to spawn a random wall (_spawn_obstacle).
          - If the wall blocks the remaining path, re-plan.
        """
        if not self.running:
            return

        # Erase the agent marker from the previous cell
        if self.agent_pos:
            pr, pc = self.agent_pos
            if self.grid_obj.grid[pr][pc] not in (START, GOAL):
                self.grid_obj.draw_cell(pr, pc, COLORS["path"])

        # Check if the agent has reached the end of the path
        if self.path_idx >= len(self.cur_path):
            self.running = False
            messagebox.showinfo(
                "Done", f"Goal reached!  Cost: {self.metrics['cost'].get()} steps")
            return

        # Move to next cell
        self.agent_pos  = self.cur_path[self.path_idx]
        self.path_idx  += 1
        ar, ac = self.agent_pos

        # Stop if we reached the goal cell
        if (ar, ac) == self.grid_obj.goal:
            self.grid_obj.draw_cell(ar, ac, COLORS[GOAL])
            self.running = False
            messagebox.showinfo(
                "Done", f"Goal reached!  Cost: {self.metrics['cost'].get()} steps")
            return

        # Draw agent at new position
        if self.grid_obj.grid[ar][ac] not in (START, GOAL):
            self.grid_obj.draw_cell(ar, ac, COLORS["agent"])

        # Try to spawn an obstacle; re-plan if path is blocked
        if self._spawn_obstacle():
            self._replan(self.agent_pos)
            return

        self.anim_id = self.root.after(ANIM_MS * 4, self._move_agent)

    def _spawn_obstacle(self):
        """
        With probability SPAWN_CHANCE, place a new wall on a random empty cell.
        Returns True if the new wall falls on the agent's remaining path.
        Returns False otherwise (no re-planning needed).
        """
        if random.random() > SPAWN_CHANCE:
            return False

        # Collect all currently empty cells
        empties = [
            (r, c)
            for r in range(self.grid_obj.rows)
            for c in range(self.grid_obj.cols)
            if self.grid_obj.grid[r][c] == EMPTY
        ]
        if not empties:
            return False

        # Pick one at random and turn it into a wall
        tr, tc = random.choice(empties)
        self.grid_obj.grid[tr][tc] = WALL
        self.grid_obj.draw_cell(tr, tc)          # show new wall immediately

        # Check if this wall is on the remaining planned path
        return (tr, tc) in self.cur_path[self.path_idx:]

    def _replan(self, new_start):
        """
        Re-run the selected algorithm from the agent's current position.
        Called immediately when a dynamic obstacle blocks the path.
        """
        # Import here to avoid circular imports at module level
        from algorithms import astar, greedy_bfs

        h_fn = self.h_fn_getter()    # get the currently selected heuristic

        if self.algo_var.get() == "A*":
            path, _, _ = astar(
                self.grid_obj.grid, new_start, self.grid_obj.goal, h_fn)
        else:
            path, _, _ = greedy_bfs(
                self.grid_obj.grid, new_start, self.grid_obj.goal, h_fn)

        if path is None:
            self.running = False
            messagebox.showwarning("Trapped", "No path from current position.")
            return

        # Update path and redraw it
        self.cur_path = path
        self.path_idx = 1                           # 0 is current cell
        self.metrics["cost"].set(str(len(path) - 1))

        for r, c in path[1:]:
            if self.grid_obj.grid[r][c] not in (START, GOAL, WALL):
                self.grid_obj.draw_cell(r, c, COLORS["path"])

        # Resume agent movement
        self.anim_id = self.root.after(ANIM_MS * 4, self._move_agent)



       
# ─────────────────────────────────────────────────────────────
# Entry point 
# FILE OVERVIEW:
#   constants.py  — colours, sizes, cell codes
#   heuristics.py — manhattan() and euclidean()
#   algorithms.py — greedy_bfs() and astar()
#   grid.py       — Grid class (canvas + mouse drawing)
#   animator.py   — Animator class (search + agent animation)
#   main.py       — App class (sidebar GUI) + entry point  ← YOU ARE HERE
# ─────────────────────────────────────────────────────────────

import tkinter as tk
from tkinter import messagebox
import time

from constants  import EMPTY, WALL, START, GOAL, DEFAULT_ROWS, DEFAULT_COLS, COLORS
from heuristics import manhattan, euclidean
from algorithms import astar, greedy_bfs
from grid       import Grid
from animator   import Animator


class App:
    """
    Main application window.
    Builds the sidebar with all controls and connects them to
    the Grid and Animator objects.
    """

    def __init__(self, root):
        self.root = root
        root.title("AI 2002 – Q6 | Dynamic Pathfinding Agent")
        root.configure(bg=COLORS["bg"])
        root.resizable(True, True)

        # ── Control variables (read by sidebar widgets) ───
        
        self.algo_var = tk.StringVar(value="A*")
        self.h_var    = tk.StringVar(value="Manhattan")
        self.mode_var = tk.StringVar(value="wall")
        self.dyn_var  = tk.BooleanVar(value=False)

        # Metric display (shown in sidebar, updated after each search)
        self.m_nodes = tk.StringVar(value="0")
        self.m_cost  = tk.StringVar(value="0")
        self.m_time  = tk.StringVar(value="0.00 ms")
        self.metrics = {"nodes": self.m_nodes,
                        "cost" : self.m_cost,
                        "time" : self.m_time}

        # Build sidebar first so we can reference its frame
        self._build_sidebar()

        # Canvas frame (right of sidebar)
        cf = tk.Frame(root, bg="#bbbbbb", bd=1, relief=tk.SUNKEN)
        cf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Grid object — owns the tkinter Canvas and the 2-D grid data
        self.grid_obj = Grid(cf, DEFAULT_ROWS, DEFAULT_COLS, self.mode_var)

        # Animator object — drives step-by-step animation and agent movement
        self.animator = Animator(
            root        = root,
            grid_obj    = self.grid_obj,
            algo_var    = self.algo_var,
            h_fn_getter = self._get_h,
            metrics     = self.metrics,
        )

   
    #  SIDEBAR CONSTRUCTION
 

    def _build_sidebar(self):
        SB = COLORS["sidebar"]

        sb = tk.Frame(self.root, bg=SB, relief=tk.RIDGE, bd=1,
                      padx=8, pady=8, width=200)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        sb.pack_propagate(False)

        # ── Tiny helper functions used only inside this method ──

        def section(text):
            tk.Frame(sb, bg="#aaaaaa", height=1).pack(fill=tk.X, pady=(8, 3))
            tk.Label(sb, text=text, bg=SB, fg="#222222",
                     font=("Arial", 8, "bold"), anchor="w").pack(fill=tk.X)

        def radio_group(var, options):
            """Create a set of radio buttons that all write to the same var."""
            for val, label in options:
                tk.Radiobutton(
                    sb, text=label, variable=var, value=val,
                    bg=SB, fg="#111111",
                    activebackground=SB, selectcolor="#cccccc",
                    font=("Arial", 9), anchor="w"
                ).pack(fill=tk.X)

        def btn(text, cmd, bg="#dddddd", fg="#000000"):
            tk.Button(sb, text=text, command=cmd, bg=bg, fg=fg,
                      relief=tk.RAISED, font=("Arial", 8, "bold"),
                      cursor="hand2").pack(fill=tk.X, pady=2)

        # Title
        tk.Label(sb, text="Pathfinding Agent", bg=SB, fg="#111111",
                 font=("Arial", 11, "bold")).pack(pady=(0, 2))
        tk.Label(sb, text="AI 2002 – Question 6", bg=SB, fg="#555555",
                 font=("Arial", 8)).pack(pady=(0, 4))

        # ── Grid size ──────
        section("Grid Size")
        sf = tk.Frame(sb, bg=SB)
        sf.pack(fill=tk.X, pady=2)
        for i, (lbl_text, attr) in enumerate([("Rows", "e_rows"),
                                               ("Cols", "e_cols")]):
            tk.Label(sf, text=f"{lbl_text}:", bg=SB,
                     font=("Arial", 8), width=5, anchor="w").grid(
                         row=i, column=0)
            e = tk.Entry(sf, width=5, font=("Arial", 8))
            e.insert(0, str(DEFAULT_ROWS if i == 0 else DEFAULT_COLS))
            e.grid(row=i, column=1, padx=2, pady=2)
            setattr(self, attr, e)
        btn("Apply Size", self._apply_size)

        # ── Algorithm selection ────────
        # Writes to self.algo_var.
        # _start() reads this to call astar() or greedy_bfs().
        section("Algorithm")
        radio_group(self.algo_var, [
            ("A*",         "A*  (optimal, slower)"),
            ("Greedy BFS", "Greedy BFS  (fast, non-optimal)"),
        ])

        # ── Heuristic selection ───────────
        # Writes to self.h_var.
        # _get_h() reads this and passes the function into the algorithm.
        section("Heuristic  h(n)")
        radio_group(self.h_var, [
            ("Manhattan", "Manhattan  |dr|+|dc|"),
            ("Euclidean", "Euclidean  sqrt(dr²+dc²)"),
        ])

        # ── Draw mode ─────────────────
        # Writes to self.mode_var.
        # Grid._apply_draw() reads this on every mouse click.
        section("Draw Mode  (click grid)")
        radio_group(self.mode_var, [
            ("wall",  "Draw Wall"),
            ("erase", "Erase"),
            ("start", "Set Start"),
            ("goal",  "Set Goal"),
        ])

        # ── Random map ─────────────────
        section("Obstacle Density")
        self.density = tk.IntVar(value=30)
        tk.Scale(sb, from_=0, to=70, orient=tk.HORIZONTAL,
                 variable=self.density, bg=SB,
                 font=("Arial", 7), highlightthickness=0).pack(fill=tk.X)
        btn("Random Map", self._random_map)

        # ── Dynamic mode ────────────────
        section("Dynamic Mode")
        tk.Checkbutton(sb, text="Enable Dynamic Obstacles",
                       variable=self.dyn_var, bg=SB, fg="#222222",
                       activebackground=SB,
                       font=("Arial", 8)).pack(anchor="w")
        tk.Label(sb,
                 text="(Walls spawn while agent moves;\nagent re-plans automatically)",
                 bg=SB, fg="#666666",
                 font=("Arial", 7), justify=tk.LEFT).pack(anchor="w")

        # ── Action buttons ────────────────
        section("Actions")
        btn("▶  Start Search", self._start,      bg="#27ae60", fg="#ffffff")
        btn("⏹  Stop",         self._stop,       bg="#e74c3c", fg="#ffffff")
        btn("Clear Path",      self._clear_path)
        btn("Clear All",       self._clear_all)

        # ── Metrics ────────────────────────
        section("Metrics")
        for label, var in [("Nodes visited:", self.m_nodes),
                            ("Path cost:",     self.m_cost),
                            ("Time:",          self.m_time)]:
            row = tk.Frame(sb, bg=SB)
            row.pack(fill=tk.X)
            tk.Label(row, text=label, bg=SB, fg="#555555",
                     font=("Arial", 8), anchor="w").pack(side=tk.LEFT)
            tk.Label(row, textvariable=var, bg=SB, fg="#000000",
                     font=("Arial", 8, "bold"),
                     anchor="e").pack(side=tk.RIGHT)

        # ── Colour legend ────────────────────
        section("Legend")
        for color, name in [
            (COLORS[START],      "Start"),
            (COLORS[GOAL],       "Goal"),
            (COLORS[WALL],       "Wall"),
            (COLORS["frontier"], "Frontier (queue)"),
            (COLORS["visited"],  "Visited"),
            (COLORS["path"],     "Final path"),
            (COLORS["agent"],    "Agent"),
        ]:
            row = tk.Frame(sb, bg=SB)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, bg=color, width=2,
                     relief=tk.SOLID, bd=1).pack(side=tk.LEFT, padx=(0, 5))
            tk.Label(row, text=name, bg=SB, fg="#222222",
                     font=("Arial", 7)).pack(side=tk.LEFT)

   
    #  GRID SIZE
    

    def _apply_size(self):
        try:
            r, c = int(self.e_rows.get()), int(self.e_cols.get())
            if not (3 <= r <= 50 and 3 <= c <= 70):
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid", "Rows: 3–50   Cols: 3–70")
            return
        self._stop()
        self.grid_obj.resize(r, c)

    def _random_map(self):
        self._stop()
        self.grid_obj.random_map(self.density.get() / 100.0)

    # ══════════════════════════════════════════════════════════
    #  HEURISTIC GETTER
    # ══════════════════════════════════════════════════════════

    def _get_h(self):
        """Return the heuristic function matching the current radio selection."""
        return manhattan if self.h_var.get() == "Manhattan" else euclidean

    # ══════════════════════════════════════════════════════════
    #  START SEARCH
    # ══════════════════════════════════════════════════════════

    def _start(self):
        self._stop()
        self._clear_path()

        g   = self.grid_obj
        h   = self._get_h()

        t0 = time.perf_counter()

        # ── CALL THE SELECTED ALGORITHM ───────────────────────
        if self.algo_var.get() == "A*":
            path, v_order, f_snaps = astar(g.grid, g.start, g.goal, h)
        else:
            path, v_order, f_snaps = greedy_bfs(g.grid, g.start, g.goal, h)
        # ─────────────────────────────────────────────────────

        ms = (time.perf_counter() - t0) * 1000
        self.m_nodes.set(str(len(v_order)))
        self.m_time.set(f"{ms:.2f} ms")

        if path is None:
            self.m_cost.set("—")
            messagebox.showwarning("No Path",
                                   "Goal is unreachable.\nTry removing some walls.")
            return

        self.m_cost.set(str(len(path) - 1))

        # Hand results to the animator
        self.animator.run(path, v_order, f_snaps,
                          dynamic_mode=self.dyn_var.get())

    # ══════════════════════════════════════════════════════════
    #  UTILITY ACTIONS
    # ══════════════════════════════════════════════════════════

    def _stop(self):
        self.animator.stop()

    def _clear_path(self):
        self.animator.stop()
        self.grid_obj.refresh_all_cells()
        self.m_nodes.set("0")
        self.m_cost.set("0")
        self.m_time.set("0.00 ms")

    def _clear_all(self):
        self._stop()
        self.grid_obj.reset()
        self.m_nodes.set("0")
        self.m_cost.set("0")
        self.m_time.set("0.00 ms")


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"+{sw // 10}+{sh // 20}")
    App(root)
    root.mainloop()











