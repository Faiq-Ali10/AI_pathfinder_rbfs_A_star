"""
Dynamic Pathfinding Agent
=========================
Controls:
  LEFT CLICK  – place/remove walls (in EDIT mode) or drag to paint
  RIGHT CLICK – remove wall
  S           – click a cell to set START
  G           – click a cell to set GOAL
  R           – generate random maze
  C           – clear grid
  SPACE       – run / stop search
  1           – A* algorithm
  2           – GBFS algorithm
  M           – toggle Manhattan / Euclidean heuristic
  D           – toggle dynamic obstacle mode
  +/-         – increase/decrease animation speed
  ESC         – quit
"""

import pygame
import sys
import heapq
import math
import random
import time

# ──────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────
ROWS, COLS    = 25, 40
CELL_SIZE     = 24
PANEL_W       = 280
WIN_W         = COLS * CELL_SIZE + PANEL_W
WIN_H         = ROWS * CELL_SIZE
FPS           = 60

# Cell states
EMPTY    = 0
WALL     = 1
START    = 2
GOAL     = 3
VISITED  = 4
FRONTIER = 5
PATH     = 6
AGENT    = 7

# Colors
C = {
    EMPTY:    (15,  17,  26),
    WALL:     (42,  45,  58),
    START:    (0,   229, 255),
    GOAL:     (255, 64,  129),
    VISITED:  (26,  35,  126),
    FRONTIER: (249, 168, 37),
    PATH:     (0,   230, 118),
    AGENT:    (255, 109, 0),
}
BG         = (8,  11,  18)
PANEL_BG   = (13, 17,  23)
BORDER     = (30, 33,  48)
TEXT_MAIN  = (224,224,224)
TEXT_DIM   = (80, 85, 100)
ACCENT     = (0, 229, 255)
ACCENT2    = (0, 230, 118)
WARN       = (255, 64, 129)

# ──────────────────────────────────────────────────
#  PRIORITY QUEUE
# ──────────────────────────────────────────────────
class MinHeap:
    def __init__(self):
        self._h = []
        self._cnt = 0
    def push(self, priority, item):
        heapq.heappush(self._h, (priority, self._cnt, item))
        self._cnt += 1
    def pop(self):
        _, _, item = heapq.heappop(self._h)
        return item
    def __len__(self):
        return len(self._h)

# ──────────────────────────────────────────────────
#  HEURISTICS
# ──────────────────────────────────────────────────
def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# ──────────────────────────────────────────────────
#  SEARCH (returns generator for animation)
# ──────────────────────────────────────────────────
DIRS = [(-1,0),(1,0),(0,-1),(0,1)]

def search_gen(grid, rows, cols, start, goal, algo, hfunc):
    """Generator that yields (visited_list, frontier_list) snapshots step by step."""
    h = hfunc
    heap = MinHeap()
    visited = set()
    g_score = {start: 0}
    parent  = {start: None}
    frontier_set = {start}

    h0 = h(start, goal)
    heap.push(h0 if algo == "gbfs" else h0, start)

    while len(heap) > 0:
        cur = heap.pop()
        if cur in visited:
            continue
        visited.add(cur)
        frontier_set.discard(cur)

        yield "step", cur, set(frontier_set), dict(parent)

        if cur == goal:
            yield "found", cur, set(frontier_set), dict(parent)
            return

        r, c = cur
        for dr, dc in DIRS:
            nr, nc = r+dr, c+dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr][nc] == WALL:
                continue
            nb = (nr, nc)
            if nb in visited:
                continue
            ng = g_score[cur] + 1
            if algo == "astar" and nb in g_score and g_score[nb] <= ng:
                continue
            g_score[nb] = ng
            parent[nb]  = cur
            prio = h(nb, goal) if algo == "gbfs" else ng + h(nb, goal)
            heap.push(prio, nb)
            frontier_set.add(nb)

    yield "no_path", None, set(), dict(parent)

def reconstruct_path(parent, goal):
    path = []
    cur  = goal
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return path

# ──────────────────────────────────────────────────
#  WALL PLACEMENT DIALOG (before simulation)
# ──────────────────────────────────────────────────

# ──────────────────────────────────────────────────
#  MAIN APP
# ──────────────────────────────────────────────────
class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Dynamic Pathfinding Agent")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock  = pygame.time.Clock()

        # Fonts
        self.font_sm  = pygame.font.SysFont("Consolas", 11)
        self.font_med = pygame.font.SysFont("Consolas", 13, bold=True)
        self.font_lg  = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_xl  = pygame.font.SysFont("Consolas", 22, bold=True)

        self.rows = ROWS
        self.cols = COLS

        # State
        self.grid      = [[EMPTY]*COLS for _ in range(ROWS)]
        self.start     = (2, 2)
        self.goal      = (ROWS-3, COLS-3)
        self.grid[self.start[0]][self.start[1]] = START
        self.grid[self.goal[0]][self.goal[1]]   = GOAL

        self.algo      = "astar"      # astar | gbfs
        self.heuristic = "manhattan"  # manhattan | euclidean
        self.dynamic   = False
        self.dyn_prob  = 0.003        # per-cell probability per frame
        self.speed     = 5            # steps per frame (1-20)

        # Edit mode: wall | set_start | set_goal
        self.edit_mode   = "wall"
        self.mouse_held  = False
        self.paint_value = WALL       # what we paint on drag

        # Animation state
        self.running      = False
        self.gen          = None
        self.visited_set  = set()
        self.frontier_set = set()
        self.path         = []
        self.agent_pos    = None
        self.agent_idx    = 0         # index along path
        self.current_path = []
        self.parent_map   = {}
        self.phase        = "search"  # search | walk

        # Metrics
        self.nodes_visited = 0
        self.path_cost     = 0
        self.elapsed_ms    = 0.0
        self.replans       = 0
        self.status        = "idle"   # idle | searching | walking | found | no_path

        self.t_start       = 0.0

        # Show wall-placement intro
        self.show_intro    = True
        self.intro_done    = False

        # Density for random maze
        self.density = 30

    # ── Grid helpers ──────────────────────────────
    def clear_overlay(self):
        """Remove VISITED / FRONTIER / PATH / AGENT overlays."""
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] in (VISITED, FRONTIER, PATH, AGENT):
                    self.grid[r][c] = EMPTY
        # Restore start/goal
        self.grid[self.start[0]][self.start[1]] = START
        self.grid[self.goal[0]][self.goal[1]]   = GOAL

    def clear_all(self):
        self.grid = [[EMPTY]*self.cols for _ in range(self.rows)]
        self.grid[self.start[0]][self.start[1]] = START
        self.grid[self.goal[0]][self.goal[1]]   = GOAL
        self.reset_state()

    def gen_maze(self):
        self.grid = [[EMPTY]*self.cols for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in (self.start, self.goal):
                    continue
                if random.random()*100 < self.density:
                    self.grid[r][c] = WALL
        self.grid[self.start[0]][self.start[1]] = START
        self.grid[self.goal[0]][self.goal[1]]   = GOAL
        self.reset_state()

    def reset_state(self):
        self.running      = False
        self.gen          = None
        self.visited_set  = set()
        self.frontier_set = set()
        self.path         = []
        self.agent_pos    = None
        self.agent_idx    = 0
        self.current_path = []
        self.parent_map   = {}
        self.phase        = "search"
        self.nodes_visited = 0
        self.path_cost    = 0
        self.elapsed_ms   = 0.0
        self.replans      = 0
        self.status       = "idle"
        self.clear_overlay()

    def cell_at(self, mx, my):
        if mx >= self.cols * CELL_SIZE:
            return None
        c = mx // CELL_SIZE
        r = my // CELL_SIZE
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return (r, c)
        return None

    # ── Start search ──────────────────────────────
    def start_search(self, from_pos=None):
        self.clear_overlay()
        pos = from_pos or self.start
        hfunc = manhattan if self.heuristic == "manhattan" else euclidean

        # Pure grid copy for search (treat START/GOAL as EMPTY for traversal)
        def cell_passable(r, c):
            return self.grid[r][c] != WALL

        self.gen          = search_gen(self.grid, self.rows, self.cols, pos, self.goal, self.algo, hfunc)
        self.visited_set  = set()
        self.frontier_set = set()
        self.phase        = "search"
        self.t_start      = time.perf_counter()
        self.status       = "searching"
        if from_pos is None:
            self.nodes_visited = 0
            self.path_cost     = 0
            self.replans       = 0
        self.running      = True

    def start_walk(self):
        self.agent_pos  = self.start
        self.agent_idx  = 0
        self.current_path = self.path[:]
        self.phase      = "walk"
        self.status     = "walking"

    # ── Update (called every frame) ───────────────
    def update(self):
        if not self.running:
            return

        if self.phase == "search":
            for _ in range(self.speed):
                if self.gen is None:
                    break
                try:
                    result = next(self.gen)
                except StopIteration:
                    self.running = False
                    break

                tag  = result[0]
                node = result[1]
                fset = result[2]
                pmap = result[3]

                if tag in ("step", "found", "no_path"):
                    self.parent_map   = pmap
                    self.frontier_set = fset

                if tag == "step":
                    self.visited_set.add(node)
                    self.nodes_visited += 1
                    # Update display
                    r, c = node
                    if self.grid[r][c] not in (START, GOAL):
                        self.grid[r][c] = VISITED
                    for (fr, fc) in fset:
                        if self.grid[fr][fc] not in (START, GOAL, VISITED):
                            self.grid[fr][fc] = FRONTIER

                elif tag == "found":
                    self.elapsed_ms = (time.perf_counter()-self.t_start)*1000
                    self.path = reconstruct_path(self.parent_map, self.goal)
                    self.path_cost = len(self.path) - 1
                    # Draw path
                    for (pr, pc) in self.path:
                        if self.grid[pr][pc] not in (START, GOAL):
                            self.grid[pr][pc] = PATH
                    self.gen = None
                    if self.dynamic:
                        self.start_walk()
                    else:
                        self.status  = "found"
                        self.running = False

                elif tag == "no_path":
                    self.elapsed_ms = (time.perf_counter()-self.t_start)*1000
                    self.status  = "no_path"
                    self.running = False
                    self.gen     = None

        elif self.phase == "walk":
            # Spawn dynamic obstacles
            if self.dynamic:
                for r in range(self.rows):
                    for c in range(self.cols):
                        if self.grid[r][c] == EMPTY:
                            if random.random() < self.dyn_prob:
                                pos = (r, c)
                                if pos not in (self.start, self.goal, self.agent_pos):
                                    self.grid[r][c] = WALL

            # Check if current path is blocked
            path_blocked = False
            for idx in range(self.agent_idx, len(self.current_path)):
                pr, pc = self.current_path[idx]
                if self.grid[pr][pc] == WALL:
                    path_blocked = True
                    break

            if path_blocked:
                # Replan from current agent position
                self.replans += 1
                hfunc = manhattan if self.heuristic == "manhattan" else euclidean
                # Run search synchronously for replan
                result = self._sync_search(self.agent_pos)
                if result:
                    self.current_path = result
                    self.agent_idx = 0
                    self.path_cost = len(result) - 1
                    # Redraw path
                    self.clear_path_overlay()
                    for (pr, pc) in self.current_path:
                        if self.grid[pr][pc] not in (START, GOAL, AGENT):
                            self.grid[pr][pc] = PATH
                else:
                    self.status  = "no_path"
                    self.running = False
                    return

            # Move agent
            for _ in range(max(1, self.speed//3)):
                if self.agent_idx >= len(self.current_path):
                    break
                nr, nc = self.current_path[self.agent_idx]
                if self.grid[nr][nc] == WALL:
                    break
                # Clear old agent pos
                if self.agent_pos:
                    ar, ac = self.agent_pos
                    if self.grid[ar][ac] == AGENT:
                        self.grid[ar][ac] = VISITED
                self.agent_pos = (nr, nc)
                ar, ac = self.agent_pos
                if self.grid[ar][ac] not in (START, GOAL):
                    self.grid[ar][ac] = AGENT
                self.agent_idx += 1
                self.elapsed_ms = (time.perf_counter()-self.t_start)*1000

                if self.agent_pos == self.goal:
                    self.grid[ar][ac] = GOAL
                    self.status  = "found"
                    self.running = False
                    break

    def clear_path_overlay(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == PATH:
                    self.grid[r][c] = VISITED

    def _sync_search(self, from_pos):
        """Run search synchronously, return path or None."""
        hfunc = manhattan if self.heuristic == "manhattan" else euclidean
        heap  = []
        heapq.heappush(heap, (0, 0, from_pos))
        visited = set()
        g_score = {from_pos: 0}
        parent  = {from_pos: None}
        cnt = 0

        while heap:
            _, _, cur = heapq.heappop(heap)
            if cur in visited:
                continue
            visited.add(cur)
            self.nodes_visited += 1
            if cur == self.goal:
                path = []
                c2 = self.goal
                while c2 is not None:
                    path.append(c2)
                    c2 = parent.get(c2)
                path.reverse()
                return path
            r, c = cur
            for dr, dc in DIRS:
                nr, nc = r+dr, c+dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                if self.grid[nr][nc] == WALL:
                    continue
                nb = (nr, nc)
                if nb in visited:
                    continue
                ng = g_score[cur] + 1
                if self.algo == "astar" and nb in g_score and g_score[nb] <= ng:
                    continue
                g_score[nb] = ng
                parent[nb]  = cur
                prio = hfunc(nb, self.goal) if self.algo == "gbfs" else ng + hfunc(nb, self.goal)
                cnt += 1
                heapq.heappush(heap, (prio, cnt, nb))
        return None

    # ── Drawing ────────────────────────────────────
    def draw_grid(self):
        for r in range(self.rows):
            for c in range(self.cols):
                state = self.grid[r][c]
                color = C.get(state, C[EMPTY])
                rect  = (c*CELL_SIZE, r*CELL_SIZE, CELL_SIZE-1, CELL_SIZE-1)
                pygame.draw.rect(self.screen, color, rect, border_radius=2)

        # Grid lines (subtle)
        for r in range(self.rows+1):
            pygame.draw.line(self.screen, BORDER, (0, r*CELL_SIZE), (self.cols*CELL_SIZE, r*CELL_SIZE))
        for c in range(self.cols+1):
            pygame.draw.line(self.screen, BORDER, (c*CELL_SIZE, 0), (c*CELL_SIZE, self.rows*CELL_SIZE))

    def draw_panel(self):
        px = self.cols * CELL_SIZE
        pygame.draw.rect(self.screen, PANEL_BG, (px, 0, PANEL_W, WIN_H))
        pygame.draw.line(self.screen, BORDER, (px, 0), (px, WIN_H), 2)

        x, y = px+14, 14
        def txt(text, color=TEXT_MAIN, font=None, dy=0):
            nonlocal y
            f = font or self.font_med
            surf = f.render(text, True, color)
            self.screen.blit(surf, (x, y+dy))
            y += surf.get_height() + 4

        def sep():
            nonlocal y
            pygame.draw.line(self.screen, BORDER, (x, y+2), (px+PANEL_W-14, y+2))
            y += 10

        def badge(label, val, color):
            nonlocal y
            lw = self.font_sm.render(label, True, TEXT_DIM)
            vw = self.font_med.render(str(val), True, color)
            self.screen.blit(lw, (x, y))
            self.screen.blit(vw, (px+PANEL_W-14-vw.get_width(), y))
            y += max(lw.get_height(), vw.get_height()) + 5

        # Title
        txt("PATHFINDING", ACCENT, self.font_xl)
        txt("AGENT", ACCENT, self.font_xl, dy=-10); y-=4
        sep()

        # Status
        status_colors = {"idle":TEXT_DIM,"searching":(249,168,37),"walking":(0,229,255),"found":(0,230,118),"no_path":(255,64,129)}
        status_text   = {"idle":"IDLE","searching":"SEARCHING...","walking":"NAVIGATING...","found":"PATH FOUND","no_path":"NO PATH!"}
        sc = status_colors.get(self.status, TEXT_DIM)
        txt(status_text.get(self.status,""), sc, self.font_lg)
        sep()

        # Metrics
        badge("Nodes Visited", self.nodes_visited, ACCENT)
        badge("Path Cost",     self.path_cost,     ACCENT2)
        badge("Time (ms)",     f"{self.elapsed_ms:.1f}", (249,168,37))
        badge("Re-plans",      self.replans,        (255,109,0))
        sep()

        # Algorithm
        txt("ALGORITHM", TEXT_DIM, self.font_sm); y+=2
        a_col = ACCENT if self.algo=="astar" else TEXT_DIM
        g_col = ACCENT if self.algo=="gbfs"  else TEXT_DIM
        txt(f"[1] A*    {'◀' if self.algo=='astar' else ''}", a_col)
        txt(f"[2] GBFS  {'◀' if self.algo=='gbfs' else ''}",  g_col)
        sep()

        txt("HEURISTIC", TEXT_DIM, self.font_sm); y+=2
        mh = ACCENT  if self.heuristic=="manhattan" else TEXT_DIM
        eh = ACCENT  if self.heuristic=="euclidean" else TEXT_DIM
        txt(f"[M] Manhattan {'◀' if self.heuristic=='manhattan' else ''}", mh)
        txt(f"[M] Euclidean {'◀' if self.heuristic=='euclidean' else ''}", eh)
        sep()

        txt("EDIT MODE", TEXT_DIM, self.font_sm); y+=2
        for key, label in [("wall","Wall Toggle"),("set_start","Set Start"),("set_goal","Set Goal")]:
            color = ACCENT if self.edit_mode==key else TEXT_DIM
            prefix = "◀ " if self.edit_mode==key else "  "
            txt(f"{prefix}{label}", color, self.font_sm)
        y+=4
        txt("[S] Set Start  [G] Set Goal", TEXT_DIM, self.font_sm)
        sep()

        txt("DYNAMIC MODE", TEXT_DIM, self.font_sm); y+=2
        dc = ACCENT2 if self.dynamic else WARN
        txt(f"[D] {'ENABLED  ← walls spawn!' if self.dynamic else 'DISABLED'}", dc)
        sep()

        txt(f"[R] Random Maze (density {self.density}%)", TEXT_DIM, self.font_sm); y+=2
        txt("[C] Clear Grid", TEXT_DIM, self.font_sm); y+=2
        txt("[SPACE] Run / Stop", TEXT_DIM, self.font_sm); y+=2
        txt("[+/-] Speed: "+str(self.speed), TEXT_DIM, self.font_sm); y+=2
        sep()

        # Legend
        txt("LEGEND", TEXT_DIM, self.font_sm); y+=4
        legend = [
            (C[START],   "Start"),
            (C[GOAL],    "Goal"),
            (C[WALL],    "Wall"),
            (C[VISITED], "Visited"),
            (C[FRONTIER],"Frontier"),
            (C[PATH],    "Path"),
            (C[AGENT],   "Agent"),
        ]
        for color, label in legend:
            pygame.draw.rect(self.screen, color, (x, y+1, 12, 12), border_radius=2)
            surf = self.font_sm.render(label, True, TEXT_DIM)
            self.screen.blit(surf, (x+18, y))
            y += 16

    def draw_intro(self):
        """Overlay telling user to place walls before starting."""
        overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        overlay.fill((8, 11, 18, 200))
        self.screen.blit(overlay, (0,0))

        lines = [
            ("DYNAMIC PATHFINDING AGENT", self.font_xl, ACCENT),
            ("", self.font_sm, TEXT_DIM),
            ("SETUP YOUR MAP BEFORE RUNNING", self.font_lg, TEXT_MAIN),
            ("", self.font_sm, TEXT_DIM),
            ("LEFT CLICK     →  Place / Remove walls",      self.font_med, TEXT_MAIN),
            ("RIGHT CLICK    →  Remove wall",               self.font_med, TEXT_MAIN),
            ("S + CLICK      →  Move Start node",           self.font_med, TEXT_MAIN),
            ("G + CLICK      →  Move Goal  node",           self.font_med, TEXT_MAIN),
            ("R              →  Generate random maze",      self.font_med, TEXT_MAIN),
            ("C              →  Clear all walls",           self.font_med, TEXT_MAIN),
            ("",self.font_sm,TEXT_DIM),
            ("SPACE          →  Begin Pathfinding",         self.font_lg,  ACCENT2),
            ("D              →  Toggle dynamic obstacles",  self.font_med, (249,168,37)),
            ("1 / 2          →  Switch A* / GBFS",          self.font_med, TEXT_MAIN),
            ("M              →  Toggle heuristic",          self.font_med, TEXT_MAIN),
            ("",self.font_sm,TEXT_DIM),
            ("Press any key or click to start editing →",   self.font_med, TEXT_DIM),
        ]
        total_h = sum(f.get_height()+6 for _, f, _ in lines)
        cy = (WIN_H - total_h) // 2
        for text, font, color in lines:
            surf = font.render(text, True, color)
            self.screen.blit(surf, ((WIN_W - surf.get_width())//2, cy))
            cy += surf.get_height() + 6

    # ── Events ─────────────────────────────────────
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if self.show_intro:
                if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                    self.show_intro = False
                return

            if event.type == pygame.KEYDOWN:
                k = event.key
                if k == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                elif k == pygame.K_SPACE:
                    if self.running:
                        self.reset_state()
                    else:
                        self.reset_state()
                        self.start_search()
                elif k == pygame.K_r:
                    self.gen_maze()
                elif k == pygame.K_c:
                    self.clear_all()
                elif k == pygame.K_1:
                    self.algo = "astar"
                elif k == pygame.K_2:
                    self.algo = "gbfs"
                elif k == pygame.K_m:
                    self.heuristic = "euclidean" if self.heuristic=="manhattan" else "manhattan"
                elif k == pygame.K_d:
                    self.dynamic = not self.dynamic
                elif k == pygame.K_s:
                    self.edit_mode = "set_start"
                elif k == pygame.K_g:
                    self.edit_mode = "set_goal"
                elif k == pygame.K_w:
                    self.edit_mode = "wall"
                elif k == pygame.K_PLUS or k == pygame.K_EQUALS:
                    self.speed = min(20, self.speed+1)
                elif k == pygame.K_MINUS:
                    self.speed = max(1, self.speed-1)

            if event.type == pygame.MOUSEBUTTONDOWN:
                self.mouse_held = True
                mx, my = event.pos
                cell = self.cell_at(mx, my)
                if cell and not self.running:
                    r, c = cell
                    if event.button == 1:
                        if self.edit_mode == "wall":
                            if self.grid[r][c] == WALL:
                                self.grid[r][c] = EMPTY
                                self.paint_value = EMPTY
                            elif self.grid[r][c] not in (START, GOAL):
                                self.grid[r][c] = WALL
                                self.paint_value = WALL
                        elif self.edit_mode == "set_start":
                            if self.grid[r][c] not in (WALL, GOAL):
                                old = self.start
                                self.grid[old[0]][old[1]] = EMPTY
                                self.start = (r, c)
                                self.grid[r][c] = START
                                self.edit_mode = "wall"
                        elif self.edit_mode == "set_goal":
                            if self.grid[r][c] not in (WALL, START):
                                old = self.goal
                                self.grid[old[0]][old[1]] = EMPTY
                                self.goal = (r, c)
                                self.grid[r][c] = GOAL
                                self.edit_mode = "wall"
                    elif event.button == 3:  # Right click = remove wall
                        if self.grid[r][c] == WALL:
                            self.grid[r][c] = EMPTY

            if event.type == pygame.MOUSEBUTTONUP:
                self.mouse_held = False

            if event.type == pygame.MOUSEMOTION:
                if self.mouse_held and not self.running and self.edit_mode == "wall":
                    mx, my = event.pos
                    cell = self.cell_at(mx, my)
                    if cell:
                        r, c = cell
                        if self.grid[r][c] not in (START, GOAL):
                            self.grid[r][c] = self.paint_value

    # ── Main loop ─────────────────────────────────
    def run(self):
        while True:
            self.clock.tick(FPS)
            self.handle_events()

            if not self.show_intro:
                self.update()

            self.screen.fill(BG)
            self.draw_grid()
            self.draw_panel()
            if self.show_intro:
                self.draw_intro()

            pygame.display.flip()


if __name__ == "__main__":
    App().run()