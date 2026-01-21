import math
import random
import numpy as np

class Node:
    __slots__ = ("x", "y", "parent")
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)
        self.parent = None

def bresenham_line(x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        yield x, y
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy

def is_free(mask_bin, x, y) -> bool:
    h, w = mask_bin.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        return False
    return mask_bin[y, x] > 0

def is_edge_free(mask_bin, x0, y0, x1, y1) -> bool:
    for x, y in bresenham_line(x0, y0, x1, y1):
        if not is_free(mask_bin, x, y):
            return False
    return True

def nearest_node(nodes, x, y):
    return min(nodes, key=lambda n: (n.x - x) ** 2 + (n.y - y) ** 2)

def steer_towards(from_node, to_x, to_y, step_size):
    angle = math.atan2(to_y - from_node.y, to_x - from_node.x)
    new_x = int(from_node.x + step_size * math.cos(angle))
    new_y = int(from_node.y + step_size * math.sin(angle))
    n = Node(new_x, new_y)
    n.parent = from_node
    return n

def run_rrt(start, goal, mask_bin,
            max_iter,
            step_size,
            goal_tol,
            bias):

    h, w = mask_bin.shape[:2]

    if not is_free(mask_bin, start[0], start[1]):
        return None

    # If goal is inside obstacle -> try to "nudge" it to a nearby free pixel
    if not is_free(mask_bin, goal[0], goal[1]):
        for r in range(1, 25):
            found = False
            for dx in (-r, 0, r):
                for dy in (-r, 0, r):
                    gx, gy = goal[0] + dx, goal[1] + dy
                    if is_free(mask_bin, gx, gy):
                        goal = (gx, gy)
                        found = True
                        break
                if found:
                    break
            if found:
                break

    nodes = [Node(start[0], start[1])]

    for _ in range(max_iter):
        if random.random() < bias:
            rx, ry = goal
        else:
            # Try to sample free space
            for _try in range(250):
                rx = random.randint(0, w - 1)
                ry = random.randint(0, h - 1)
                if is_free(mask_bin, rx, ry):
                    break

        nearest = nearest_node(nodes, rx, ry)
        new_node = steer_towards(nearest, rx, ry, step_size)
        new_node.x = max(0, min(new_node.x, w - 1))
        new_node.y = max(0, min(new_node.y, h - 1))

        if is_edge_free(mask_bin, nearest.x, nearest.y, new_node.x, new_node.y):
            nodes.append(new_node)
            if math.hypot(new_node.x - goal[0], new_node.y - goal[1]) <= goal_tol:
                goal_node = Node(goal[0], goal[1])
                goal_node.parent = new_node
                return goal_node

    return None

def build_path(goal_node):
    if goal_node is None:
        return []
    path = []
    n = goal_node
    while n is not None:
        path.append((n.x, n.y))
        n = n.parent
    return path[::-1]

def closest_path_idx(user_xy, path) -> int:
    if not path:
        return 0
    ux, uy = user_xy
    d = [math.hypot(ux - px, uy - py) for px, py in path]
    return int(np.argmin(d))

def user_off_path(user_xy, path, tol) -> bool:
    if not path:
        return True
    ux, uy = user_xy
    dmin = min(math.hypot(ux - px, uy - py) for px, py in path)
    return dmin > tol

def path_ok_from(path, start_idx, mask_bin) -> bool:
    if not path:
        return False
    start_idx = max(0, min(start_idx, len(path) - 1))
    rem = path[start_idx:]
    if len(rem) < 2:
        return True
    for i in range(len(rem) - 1):
        if not is_edge_free(mask_bin, rem[i][0], rem[i][1], rem[i + 1][0], rem[i + 1][1]):
            return False
    return True
