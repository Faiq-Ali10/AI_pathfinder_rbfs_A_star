# Dynamic Pathfinding Agent

An interactive, real-time visualization tool built with **Pygame** to demonstrate pathfinding algorithms in static and dynamic environments. This application allows users to build custom mazes and observe how agents navigate through them using A* or Greedy Best-First Search.



## 🚀 Features

* **Real-time Visualization:** Watch the "frontier" expand and "visited" nodes populate as the algorithm searches for the goal.
* **Dual Algorithms:**
    * **A* (A-Star):** Guarantees the shortest path by balancing cost-to-reach (g) and estimated distance to goal (h).
    * **GBFS (Greedy Best-First Search):** Prioritizes speed by only looking at the estimated distance to the goal.
* **Dynamic Obstacle Mode:** Enable a "live" environment where walls randomly appear while the agent is moving, forcing the agent to **re-plan** its route on the fly.
* **Custom Map Editing:** Draw your own walls, move the start/goal points, or generate a random maze.
* **Adjustable Heuristics:** Toggle between **Manhattan** and **Euclidean** distance formulas.

---

## 🎮 Controls

### Map Construction
| Key | Action |
| :--- | :--- |
| **LEFT CLICK** | Place/Remove walls (Click or Drag to paint) |
| **RIGHT CLICK** | Specifically remove walls |
| **S** | Click a cell to move the **Start** node (Cyan) |
| **G** | Click a cell to move the **Goal** node (Pink) |
| **R** | Generate a **Random Maze** (30% density) |
| **C** | **Clear** all walls and paths |

### Simulation
| Key | Action |
| :--- | :--- |
| **SPACE** | Start / Stop / Reset the search |
| **1 / 2** | Switch between **A\*** and **GBFS** algorithms |
| **M** | Toggle Heuristic (**Manhattan / Euclidean**) |
| **D** | Toggle **Dynamic Mode** (Obstacles spawn during walk) |
| **+ / -** | Increase or decrease the animation speed |
| **ESC** | Quit application |

---

## 🛠️ Technical Details

### Pathfinding Logic
The agent uses a **Priority Queue (Min-Heap)** to efficiently select the next best node to explore.



* **A\* Scoring:** $f(n) = g(n) + h(n)$
    * *Where $g(n)$ is the actual cost from start to $n$, and $h(n)$ is the heuristic estimate to goal.*
* **GBFS Scoring:** $f(n) = h(n)$
    * *Prioritizes nodes that appear closest to the goal, regardless of path cost.*

### Requirements
* Python 3.x
* Pygame library

```bash
pip install pygame