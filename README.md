# AI 2002 - Assignment 2 | Question 6
## Dynamic Pathfinding Agent

---

## How to Run

```bash
python pathfinding_agent.py
```

**Requirements:** Python 3.x only — no external libraries needed.  
`tkinter` is bundled with Python on Windows and most Linux distros.

> If tkinter is missing on Linux:
> ```bash
> sudo apt-get install python3-tk
> ```

---

## Features

| Feature | Description |
|---|---|
| **Grid Editor** | Click/drag to draw walls, set start & goal |
| **Random Maps** | Generate mazes with adjustable obstacle density |
| **A\* Search** | Optimal path, f(n) = g(n) + h(n) |
| **Greedy BFS** | Fast but non-optimal, f(n) = h(n) |
| **Manhattan Distance** | Best for 4-directional grids |
| **Euclidean Distance** | Straight-line distance heuristic |
| **Dynamic Mode** | Random obstacles spawn while agent moves; agent re-plans instantly |
| **Animation** | Yellow = frontier, Blue = visited, Green = final path, Purple = agent |
| **Metrics** | Nodes visited, path cost, execution time |

---

## How to Use

1. **Set Grid Size** — Enter rows/cols and click "Apply Grid Size"
2. **Draw Walls** — Select "Draw Wall" and click/drag on the grid
3. **Set Start/Goal** — Use the draw mode radio buttons
4. **Choose Algorithm** — A* (optimal) or Greedy BFS (fast)
5. **Choose Heuristic** — Manhattan or Euclidean
6. **Click "Start Search"** — Watch the animation
7. **Dynamic Mode** — Enable the checkbox; agent will re-plan when obstacles appear

---

## Algorithm Summary

### A* Search
- Uses `f(n) = g(n) + h(n)`
- Always finds the shortest path (optimal) when heuristic is admissible
- Expands more nodes but guarantees correctness

### Greedy Best-First Search
- Uses `f(n) = h(n)` only — ignores actual path cost
- Faster but may find a longer path
- Can get stuck behind obstacles

---

## File Structure

```
pathfinding_agent.py    ← Single file, all code with comments
README.md               ← This file
```
