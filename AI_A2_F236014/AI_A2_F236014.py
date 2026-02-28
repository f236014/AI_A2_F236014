import tkinter as tk
from tkinter import messagebox, ttk
import heapq
import math
import time
import random

# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────

CELL_SIZE    = 26
DEFAULT_ROWS = 20
DEFAULT_COLS = 28
ANIM_MS      = 30
SPAWN_CHANCE = 0.04

EMPTY = 0
WALL  = 1
START = 2
GOAL  = 3

COLORS = {
    EMPTY      : "#ffffff",
    WALL       : "#2d2d2d",
    START      : "#27ae60",
    GOAL       : "#e74c3c",
    "frontier" : "#f39c12",
    "visited"  : "#3498db",
    "path"     : "#2ecc71",
    "agent"    : "#9b59b6",
    "grid"     : "#cccccc",
    "bg"       : "#f0f2f5",
    "sidebar"  : "#ffffff",
}

# ─────────────────────────────────────────────────────────────
#  HEURISTIC FUNCTIONS
# ─────────────────────────────────────────────────────────────

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# ─────────────────────────────────────────────────────────────
#  SHARED HELPERS
# ─────────────────────────────────────────────────────────────

def get_neighbors(node, rows, cols, grid):
    r, c = node
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != WALL:
            yield (nr, nc)

def build_path(came_from, goal):
    path, cur = [], goal
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    return path

# ─────────────────────────────────────────────────────────────
#  ALGORITHM 1 — GREEDY BEST-FIRST SEARCH
# ─────────────────────────────────────────────────────────────

def greedy_bfs(grid, start, goal, h_fn):
    rows, cols = len(grid), len(grid[0])
    pq = [(h_fn(start, goal), start)]
    came_from = {start: None}
    visited   = {start}
    v_order   = []
    f_snaps   = []

    while pq:
        _, cur = heapq.heappop(pq)
        v_order.append(cur)
        if cur == goal:
            return build_path(came_from, goal), v_order, f_snaps
        for nb in get_neighbors(cur, rows, cols, grid):
            if nb not in visited:
                visited.add(nb)
                came_from[nb] = cur
                heapq.heappush(pq, (h_fn(nb, goal), nb))
        f_snaps.append([x[1] for x in pq])

    return None, v_order, f_snaps

# ─────────────────────────────────────────────────────────────
#  ALGORITHM 2 — A* SEARCH
# ─────────────────────────────────────────────────────────────

def astar(grid, start, goal, h_fn):
    rows, cols = len(grid), len(grid[0])
    g = {start: 0}
    pq = [(h_fn(start, goal), 0, start)]
    came_from = {start: None}
    expanded  = set()
    v_order   = []
    f_snaps   = []

    while pq:
        _, g_cur, cur = heapq.heappop(pq)
        if cur in expanded:
            continue
        expanded.add(cur)
        v_order.append(cur)
        if cur == goal:
            return build_path(came_from, goal), v_order, f_snaps
        for nb in get_neighbors(cur, rows, cols, grid):
            new_g = g[cur] + 1
            if nb not in g or new_g < g[nb]:
                g[nb] = new_g
                came_from[nb] = cur
                heapq.heappush(pq, (new_g + h_fn(nb, goal), new_g, nb))
        f_snaps.append([x[2] for x in pq])

    return None, v_order, f_snaps

# ─────────────────────────────────────────────────────────────
#  GUI APPLICATION
# ─────────────────────────────────────────────────────────────

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("AI 2002 – Q6  |  Dynamic Pathfinding Agent")
        self.root.configure(bg=COLORS["bg"])
        self.root.resizable(True, True)

        self.rows  = DEFAULT_ROWS
        self.cols  = DEFAULT_COLS
        self.grid  = []
        self.start = (0, 0)
        self.goal  = (self.rows-1, self.cols-1)

        self.algo_var = tk.StringVar(value="A*")
        self.h_var    = tk.StringVar(value="Manhattan")
        self.mode_var = tk.StringVar(value="wall")
        self.dyn_var  = tk.BooleanVar(value=False)

        self.lbl_nodes = tk.StringVar(value="0")
        self.lbl_cost  = tk.StringVar(value="0")
        self.lbl_time  = tk.StringVar(value="0 ms")

        self._nodes_so_far    = 0
        self._search_start    = 0.0
        self._timer_id        = None
        self._total_path_cost = 0

        self.running   = False
        self.anim_id   = None
        self.agent_pos = None
        self.cur_path  = []
        self.path_idx  = 0

        self._build_ui()
        self._reset_grid()
        self._redraw()

    # ══════════════════════════════════════════════════════════
    #  BUILD UI
    # ══════════════════════════════════════════════════════════

    def _build_ui(self):
        # ── Outer layout: sidebar | canvas ────────────────────
        outer = tk.Frame(self.root, bg=COLORS["bg"])
        outer.pack(fill=tk.BOTH, expand=True)

        # ── Sidebar with scrollbar ─────────────────────────────
        sb_container = tk.Frame(outer, bg=COLORS["bg"])
        sb_container.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0), pady=10)

        sb_canvas = tk.Canvas(sb_container, bg=COLORS["sidebar"],
                              highlightthickness=0, width=230)
        scrollbar = ttk.Scrollbar(sb_container, orient="vertical",
                                  command=sb_canvas.yview)
        sb_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        sb_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sb = tk.Frame(sb_canvas, bg=COLORS["sidebar"])
        sb_window = sb_canvas.create_window((0, 0), window=sb, anchor="nw")

        def _on_frame_configure(e):
            sb_canvas.configure(scrollregion=sb_canvas.bbox("all"))
        def _on_canvas_configure(e):
            sb_canvas.itemconfig(sb_window, width=e.width)

        sb.bind("<Configure>", _on_frame_configure)
        sb_canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(e):
            sb_canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        sb_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # ── Sidebar padding wrapper ────────────────────────────
        pad = tk.Frame(sb, bg=COLORS["sidebar"])
        pad.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        # ── Helper builders ────────────────────────────────────
        def section_title(text):
            tk.Label(pad, text=text.upper(),
                     bg=COLORS["sidebar"], fg="#888888",
                     font=("Arial", 7, "bold"),
                     anchor="w").pack(fill=tk.X, pady=(12, 2))
            tk.Frame(pad, bg="#e0e0e0", height=1).pack(fill=tk.X)

        def card(inner_fn):
            """Wrap content in a subtle card frame."""
            f = tk.Frame(pad, bg="#f7f8fa", relief=tk.FLAT,
                         highlightbackground="#e0e0e0",
                         highlightthickness=1)
            f.pack(fill=tk.X, pady=(4, 2))
            inner_fn(f)
            return f

        def radio_group(parent, var, options):
            for val, txt in options:
                tk.Radiobutton(parent, text=txt, variable=var, value=val,
                               bg=parent["bg"], fg="#333333",
                               activebackground=parent["bg"],
                               selectcolor="#d0e8ff",
                               font=("Arial", 9), anchor="w",
                               padx=6).pack(fill=tk.X, pady=1)

        def action_btn(text, cmd, bg="#eeeeee", fg="#111111", bold=False):
            tk.Button(pad, text=text, command=cmd,
                      bg=bg, fg=fg,
                      relief=tk.FLAT,
                      font=("Arial", 9, "bold" if bold else "normal"),
                      cursor="hand2",
                      activebackground=bg,
                      padx=8, pady=5).pack(fill=tk.X, pady=2)

        # ── Title ──────────────────────────────────────────────
        tk.Label(pad, text="Pathfinding Agent",
                 bg=COLORS["sidebar"], fg="#111111",
                 font=("Arial", 13, "bold"), anchor="w").pack(fill=tk.X)
        tk.Label(pad, text="AI 2002 – Question 6",
                 bg=COLORS["sidebar"], fg="#888888",
                 font=("Arial", 8), anchor="w").pack(fill=tk.X, pady=(0, 4))

        # ── Grid Size ──────────────────────────────────────────
        section_title("Grid Size")
        def _grid_size_body(f):
            inner = tk.Frame(f, bg=f["bg"], padx=8, pady=6)
            inner.pack(fill=tk.X)
            for i, (lbl_text, entry_attr, default) in enumerate([
                ("Rows", "e_rows", str(DEFAULT_ROWS)),
                ("Cols", "e_cols", str(DEFAULT_COLS)),
            ]):
                row = tk.Frame(inner, bg=f["bg"])
                row.pack(fill=tk.X, pady=2)
                tk.Label(row, text=lbl_text, bg=f["bg"], fg="#555",
                         font=("Arial", 8), width=5, anchor="w").pack(side=tk.LEFT)
                e = tk.Entry(row, width=6, font=("Arial", 9),
                             relief=tk.SOLID, bd=1)
                e.insert(0, default)
                e.pack(side=tk.LEFT, padx=(4, 0))
                setattr(self, entry_attr, e)
            tk.Button(inner, text="Apply Size", command=self._apply_size,
                      bg="#e8e8e8", fg="#222", relief=tk.FLAT,
                      font=("Arial", 8), cursor="hand2",
                      padx=6, pady=3).pack(fill=tk.X, pady=(6,0))
        card(_grid_size_body)

        # ── Algorithm ─────────────────────────────────────────
        section_title("Algorithm")
        def _algo_body(f):
            radio_group(f, self.algo_var, [
                ("A*",         "A*  (optimal, slower)"),
                ("Greedy BFS", "Greedy BFS  (fast, non-optimal)"),
            ])
        card(_algo_body)

        # ── Heuristic ─────────────────────────────────────────
        section_title("Heuristic  h(n)")
        def _heur_body(f):
            radio_group(f, self.h_var, [
                ("Manhattan", "Manhattan  |dr|+|dc|"),
                ("Euclidean", "Euclidean  √(dr²+dc²)"),
            ])
        card(_heur_body)

        # ── Draw Mode ─────────────────────────────────────────
        section_title("Draw Mode  (click grid)")
        def _draw_body(f):
            for val, txt, dot_color in [
                ("wall",  "Draw Wall",  "#2d2d2d"),
                ("erase", "Erase",      "#ffffff"),
                ("start", "Set Start",  "#27ae60"),
                ("goal",  "Set Goal",   "#e74c3c"),
            ]:
                row = tk.Frame(f, bg=f["bg"])
                row.pack(fill=tk.X, pady=1)
                tk.Radiobutton(row, text=txt, variable=self.mode_var, value=val,
                               bg=f["bg"], fg="#333333",
                               activebackground=f["bg"],
                               selectcolor="#d0e8ff",
                               font=("Arial", 9), anchor="w",
                               padx=6).pack(side=tk.LEFT)
                tk.Label(row, bg=dot_color, width=2,
                         relief=tk.SOLID, bd=1).pack(side=tk.RIGHT, padx=6)
        card(_draw_body)

        # ── Random Map ────────────────────────────────────────
        section_title("Random Map")
        def _rand_body(f):
            inner = tk.Frame(f, bg=f["bg"], padx=8, pady=4)
            inner.pack(fill=tk.X)
            self.density = tk.IntVar(value=30)
            hdr = tk.Frame(inner, bg=f["bg"])
            hdr.pack(fill=tk.X)
            tk.Label(hdr, text="Obstacle Density", bg=f["bg"], fg="#555",
                     font=("Arial", 8), anchor="w").pack(side=tk.LEFT)
            self._density_lbl = tk.Label(hdr, text="30%", bg=f["bg"], fg="#3498db",
                                         font=("Arial", 8, "bold"))
            self._density_lbl.pack(side=tk.RIGHT)

            def _update_density_lbl(val):
                self._density_lbl.config(text=f"{int(float(val))}%")

            tk.Scale(inner, from_=0, to=70, orient=tk.HORIZONTAL,
                     variable=self.density, bg=f["bg"],
                     font=("Arial", 7), highlightthickness=0,
                     troughcolor="#d0d0d0", sliderrelief=tk.FLAT,
                     command=_update_density_lbl,
                     showvalue=False).pack(fill=tk.X, pady=(2,4))
            tk.Button(inner, text="🎲  Random Map", command=self._random_map,
                      bg="#e8e8e8", fg="#222", relief=tk.FLAT,
                      font=("Arial", 8), cursor="hand2",
                      padx=6, pady=3).pack(fill=tk.X)
        card(_rand_body)

        # ── Dynamic Mode ──────────────────────────────────────
        section_title("Dynamic Mode")
        def _dyn_body(f):
            inner = tk.Frame(f, bg=f["bg"], padx=8, pady=6)
            inner.pack(fill=tk.X)
            tk.Checkbutton(inner, text="Enable Dynamic Obstacles",
                           variable=self.dyn_var, bg=f["bg"],
                           fg="#333333", activebackground=f["bg"],
                           font=("Arial", 9), anchor="w").pack(anchor="w")
            tk.Label(inner,
                     text="Obstacles spawn mid-run;\nagent re-plans automatically.",
                     bg=f["bg"], fg="#999999",
                     font=("Arial", 7), justify=tk.LEFT).pack(anchor="w", pady=(2,0))
        card(_dyn_body)

        # ── Actions ───────────────────────────────────────────
        section_title("Actions")
        action_btn("▶  Start Search", self._start,   bg="#27ae60", fg="#ffffff", bold=True)
        action_btn("⏹  Stop",         self._stop,    bg="#e74c3c", fg="#ffffff")
        action_btn("↺  Clear Path",   self._clear_path)
        action_btn("✕  Clear All",    self._clear_all)

        # ── Metrics Dashboard ─────────────────────────────────
        section_title("Real-Time Metrics")
        dash = tk.Frame(pad, bg="#1a252f", relief=tk.FLAT,
                        highlightbackground="#0d1b24",
                        highlightthickness=1)
        dash.pack(fill=tk.X, pady=(4, 2))

        metrics = [
            ("Nodes Visited", self.lbl_nodes, "#f39c12", "⬡"),
            ("Path Cost",     self.lbl_cost,  "#2ecc71", "↗"),
            ("Exec Time",     self.lbl_time,  "#3498db", "⏱"),
        ]
        for i, (label_text, var, val_color, icon) in enumerate(metrics):
            row_f = tk.Frame(dash, bg="#1a252f")
            row_f.pack(fill=tk.X, padx=10, pady=(8 if i == 0 else 4, 4))

            # icon + label row
            top = tk.Frame(row_f, bg="#1a252f")
            top.pack(fill=tk.X)
            tk.Label(top, text=icon, bg="#1a252f", fg=val_color,
                     font=("Arial", 9)).pack(side=tk.LEFT)
            tk.Label(top, text=f"  {label_text}",
                     bg="#1a252f", fg="#7f8c8d",
                     font=("Arial", 7, "bold")).pack(side=tk.LEFT)

            # value
            tk.Label(row_f, textvariable=var,
                     bg="#1a252f", fg=val_color,
                     font=("Arial", 16, "bold"),
                     anchor="w").pack(anchor="w")

            if i < len(metrics) - 1:
                tk.Frame(dash, bg="#2c3e50", height=1).pack(fill=tk.X, padx=10)

        tk.Frame(dash, height=6, bg="#1a252f").pack()  # bottom padding

        # ── Legend ────────────────────────────────────────────
        section_title("Legend")
        legend_items = [
            (COLORS[START],      "Start"),
            (COLORS[GOAL],       "Goal"),
            (COLORS[WALL],       "Wall"),
            (COLORS["frontier"], "Frontier (queue)"),
            (COLORS["visited"],  "Visited"),
            (COLORS["path"],     "Final path"),
            (COLORS["agent"],    "Agent"),
        ]
        legend_frame = tk.Frame(pad, bg="#f7f8fa",
                                highlightbackground="#e0e0e0",
                                highlightthickness=1)
        legend_frame.pack(fill=tk.X, pady=(4, 10))

        for i, (color, name) in enumerate(legend_items):
            row = tk.Frame(legend_frame, bg="#f7f8fa")
            row.pack(fill=tk.X, padx=8, pady=2)
            tk.Label(row, bg=color, width=3,
                     relief=tk.SOLID, bd=1).pack(side=tk.LEFT, padx=(0,8))
            tk.Label(row, text=name, bg="#f7f8fa",
                     fg="#333333", font=("Arial", 8)).pack(side=tk.LEFT, anchor="w")

        # ── Canvas area ────────────────────────────────────────
        canvas_outer = tk.Frame(outer, bg=COLORS["bg"])
        canvas_outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                          padx=10, pady=10)

        # Canvas header
        header = tk.Frame(canvas_outer, bg=COLORS["bg"])
        header.pack(fill=tk.X, pady=(0, 6))
        tk.Label(header, text="Grid",
                 bg=COLORS["bg"], fg="#444444",
                 font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        tk.Label(header,
                 text="Left-click: draw  |  Right-click: erase  |  Drag to paint",
                 bg=COLORS["bg"], fg="#aaaaaa",
                 font=("Arial", 8)).pack(side=tk.LEFT, padx=12)

        cf = tk.Frame(canvas_outer, bg="#cccccc", bd=1, relief=tk.SOLID)
        cf.pack(anchor="nw")

        self.canvas = tk.Canvas(cf,
                                width  = self.cols * CELL_SIZE,
                                height = self.rows * CELL_SIZE,
                                bg=COLORS[EMPTY], highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>",  self._click)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<Button-3>",  self._rclick)

    # ══════════════════════════════════════════════════════════
    #  GRID
    # ══════════════════════════════════════════════════════════

    def _reset_grid(self):
        self.grid  = [[EMPTY]*self.cols for _ in range(self.rows)]
        self.start = (0, 0)
        self.goal  = (self.rows-1, self.cols-1)
        self.grid[0][0]                     = START
        self.grid[self.rows-1][self.cols-1] = GOAL

    def _apply_size(self):
        try:
            r, c = int(self.e_rows.get()), int(self.e_cols.get())
            if not (3<=r<=50 and 3<=c<=70): raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Rows: 3–50   Cols: 3–70")
            return
        self._stop()
        self.rows, self.cols = r, c
        self.canvas.config(width=c*CELL_SIZE, height=r*CELL_SIZE)
        self._reset_grid()
        self._redraw()

    def _redraw(self):
        self.canvas.delete("all")
        for r in range(self.rows):
            for c in range(self.cols):
                self._cell(r, c)

    def _cell(self, r, c, override=None):
        x1, y1 = c*CELL_SIZE, r*CELL_SIZE
        color   = override if override else COLORS[self.grid[r][c]]
        self.canvas.create_rectangle(x1, y1, x1+CELL_SIZE, y1+CELL_SIZE,
                                     fill=color, outline=COLORS["grid"], width=1)

    # ══════════════════════════════════════════════════════════
    #  MOUSE INPUT
    # ══════════════════════════════════════════════════════════

    def _to_cell(self, e):
        return (max(0,min(e.y//CELL_SIZE, self.rows-1)),
                max(0,min(e.x//CELL_SIZE, self.cols-1)))

    def _click(self, e):
        if not self.running: self._draw(*self._to_cell(e))

    def _drag(self, e):
        if not self.running: self._draw(*self._to_cell(e))

    def _rclick(self, e):
        if self.running: return
        r, c = self._to_cell(e)
        if self.grid[r][c] not in (START, GOAL):
            self.grid[r][c] = EMPTY
            self._cell(r, c)

    def _draw(self, r, c):
        mode = self.mode_var.get()
        cell = self.grid[r][c]
        if mode == "wall":
            if cell in (START, GOAL): return
            self.grid[r][c] = WALL;  self._cell(r, c)
        elif mode == "erase":
            if cell in (START, GOAL): return
            self.grid[r][c] = EMPTY; self._cell(r, c)
        elif mode == "start":
            sr, sc = self.start
            self.grid[sr][sc] = EMPTY; self._cell(sr, sc)
            self.start = (r, c)
            self.grid[r][c] = START;   self._cell(r, c)
        elif mode == "goal":
            gr, gc = self.goal
            self.grid[gr][gc] = EMPTY; self._cell(gr, gc)
            self.goal = (r, c)
            self.grid[r][c] = GOAL;    self._cell(r, c)

    # ══════════════════════════════════════════════════════════
    #  RANDOM MAP
    # ══════════════════════════════════════════════════════════

    def _random_map(self):
        self._stop()
        d = self.density.get() / 100.0
        self._reset_grid()
        for r in range(self.rows):
            for c in range(self.cols):
                if (r,c) not in (self.start, self.goal):
                    if random.random() < d:
                        self.grid[r][c] = WALL
        self._redraw()

    # ══════════════════════════════════════════════════════════
    #  LIVE TIMER
    # ══════════════════════════════════════════════════════════

    def _start_timer(self):
        self._search_start = time.perf_counter()
        self._tick_timer()

    def _tick_timer(self):
        if not self.running:
            return
        elapsed_ms = (time.perf_counter() - self._search_start) * 1000
        self.lbl_time.set(f"{elapsed_ms:.0f} ms")
        self._timer_id = self.root.after(50, self._tick_timer)

    def _stop_timer(self):
        if self._timer_id:
            self.root.after_cancel(self._timer_id)
            self._timer_id = None
        elapsed_ms = (time.perf_counter() - self._search_start) * 1000
        self.lbl_time.set(f"{elapsed_ms:.0f} ms")

    # ══════════════════════════════════════════════════════════
    #  SEARCH
    # ══════════════════════════════════════════════════════════

    def _get_h(self):
        return manhattan if self.h_var.get() == "Manhattan" else euclidean

    def _start(self):
        self._stop()
        self._clear_path()
        h = self._get_h()

        self._nodes_so_far = 0
        self.lbl_nodes.set("0")
        self.lbl_cost.set("—")
        self.lbl_time.set("0 ms")

        if self.algo_var.get() == "A*":
            path, v_order, f_snaps = astar(self.grid, self.start, self.goal, h)
        else:
            path, v_order, f_snaps = greedy_bfs(self.grid, self.start, self.goal, h)

        if path is None:
            self.lbl_nodes.set(str(len(v_order)))
            self.lbl_cost.set("No path")
            self.lbl_time.set("—")
            messagebox.showwarning("No Path", "Goal unreachable. Remove some walls.")
            return

        self._total_path_cost = len(path) - 1
        self.cur_path  = path
        self.agent_pos = self.start
        self.path_idx  = 0
        self.running   = True

        self._start_timer()
        self._anim(v_order, f_snaps, path, step=0)

    # ══════════════════════════════════════════════════════════
    #  ANIMATION
    # ══════════════════════════════════════════════════════════

    def _anim(self, v_order, f_snaps, path, step):
        if not self.running:
            return

        if step < len(v_order):
            r, c = v_order[step]
            if self.grid[r][c] not in (START, GOAL):
                self._cell(r, c, COLORS["visited"])

            if step < len(f_snaps):
                for fr, fc in f_snaps[step]:
                    if self.grid[fr][fc] not in (START, GOAL, WALL):
                        self._cell(fr, fc, COLORS["frontier"])

            self._nodes_so_far += 1
            self.lbl_nodes.set(str(self._nodes_so_far))

            self.anim_id = self.root.after(
                ANIM_MS, lambda: self._anim(v_order, f_snaps, path, step+1))

        else:
            for r, c in path:
                if self.grid[r][c] not in (START, GOAL):
                    self._cell(r, c, COLORS["path"])

            self.lbl_cost.set(str(self._total_path_cost))

            if self.dyn_var.get():
                self.path_idx = 0
                self.anim_id  = self.root.after(ANIM_MS*3, self._move)
            else:
                self.running = False
                self._stop_timer()

    # ══════════════════════════════════════════════════════════
    #  AGENT MOVEMENT
    # ══════════════════════════════════════════════════════════

    def _move(self):
        if not self.running:
            return

        if self.agent_pos:
            pr, pc = self.agent_pos
            if self.grid[pr][pc] not in (START, GOAL):
                self._cell(pr, pc, COLORS["path"])

        if self.path_idx >= len(self.cur_path):
            self.running = False
            self._stop_timer()
            messagebox.showinfo("Done", f"Goal reached!  Cost: {self.lbl_cost.get()} steps")
            return

        self.agent_pos  = self.cur_path[self.path_idx]
        self.path_idx  += 1
        ar, ac = self.agent_pos

        if (ar, ac) == self.goal:
            self._cell(ar, ac, COLORS[GOAL])
            self.running = False
            self._stop_timer()
            messagebox.showinfo("Done", f"Goal reached!  Cost: {self.lbl_cost.get()} steps")
            return

        if self.grid[ar][ac] not in (START, GOAL):
            self._cell(ar, ac, COLORS["agent"])

        if self._spawn_obstacle():
            self._replan(self.agent_pos)
            return

        self.anim_id = self.root.after(ANIM_MS*4, self._move)

    def _spawn_obstacle(self):
        if random.random() > SPAWN_CHANCE:
            return False
        empties = [(r,c) for r in range(self.rows)
                         for c in range(self.cols)
                         if self.grid[r][c] == EMPTY]
        if not empties:
            return False
        tr, tc = random.choice(empties)
        self.grid[tr][tc] = WALL
        self._cell(tr, tc)
        return (tr, tc) in self.cur_path[self.path_idx:]

    def _replan(self, new_start):
        h = self._get_h()
        if self.algo_var.get() == "A*":
            path, _, _ = astar(self.grid, new_start, self.goal, h)
        else:
            path, _, _ = greedy_bfs(self.grid, new_start, self.goal, h)

        if path is None:
            self.running = False
            self._stop_timer()
            messagebox.showwarning("Trapped", "No path from current position.")
            return

        self.cur_path = path
        self.path_idx = 1
        self.lbl_cost.set(str(len(path) - 1))
        for r, c in path[1:]:
            if self.grid[r][c] not in (START, GOAL, WALL):
                self._cell(r, c, COLORS["path"])
        self.anim_id = self.root.after(ANIM_MS*4, self._move)

    # ══════════════════════════════════════════════════════════
    #  UTILITY
    # ══════════════════════════════════════════════════════════

    def _stop(self):
        self.running = False
        self._stop_timer()
        if self.anim_id:
            self.root.after_cancel(self.anim_id)
            self.anim_id = None

    def _clear_path(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self._cell(r, c)
        self._nodes_so_far = 0
        self.lbl_nodes.set("0")
        self.lbl_cost.set("0")
        self.lbl_time.set("0 ms")

    def _clear_all(self):
        self._stop()
        self._reset_grid()
        self._redraw()
        self._nodes_so_far = 0
        self.lbl_nodes.set("0")
        self.lbl_cost.set("0")
        self.lbl_time.set("0 ms")

# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    root.update_idletasks()
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"+{w//10}+{h//20}")
    App(root)
    root.mainloop()