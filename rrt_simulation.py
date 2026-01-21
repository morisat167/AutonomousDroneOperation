import argparse
import math
import random
import cv2
import numpy as np


class Node:
    __slots__ = ("x", "y", "parent")
    def __init__(self, x, y, parent=None):
        self.x = int(x)
        self.y = int(y)
        self.parent = parent


def is_free(free, x, y):
    h, w = free.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        return False
    return free[y, x] != 0


def collision_free_segment(free, x1, y1, x2, y2, step=2):
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist == 0:
        return is_free(free, x1, y1)
    n = max(1, int(dist / step))
    for i in range(n + 1):
        t = i / n
        x = int(round(x1 + t * dx))
        y = int(round(y1 + t * dy))
        if not is_free(free, x, y):
            return False
    return True


def nearest(nodes, x, y):
    best = None
    best_d = 1e18
    for nd in nodes:
        d = (nd.x - x) ** 2 + (nd.y - y) ** 2
        if d < best_d:
            best_d = d
            best = nd
    return best


def steer(x1, y1, x2, y2, step):
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist <= step:
        return int(x2), int(y2)
    ux = dx / dist
    uy = dy / dist
    return int(round(x1 + ux * step)), int(round(y1 + uy * step))


def rrt(free, start, goal, max_iter=6000, step=18, goal_bias=0.15, goal_tol=12):
    sx, sy = start
    gx, gy = goal
    h, w = free.shape[:2]

    if not is_free(free, sx, sy) or not is_free(free, gx, gy):
        return None, []

    nodes = [Node(sx, sy, None)]

    for _ in range(max_iter):
        # Sample
        if random.random() < goal_bias:
            rx, ry = gx, gy
        else:
            rx = random.randint(0, w - 1)
            ry = random.randint(0, h - 1)

        nn = nearest(nodes, rx, ry)
        nx, ny = steer(nn.x, nn.y, rx, ry, step)

        if not is_free(free, nx, ny):
            continue
        if not collision_free_segment(free, nn.x, nn.y, nx, ny):
            continue

        new_node = Node(nx, ny, nn)
        nodes.append(new_node)

        # Try connect to goal
        if (nx - gx) ** 2 + (ny - gy) ** 2 <= goal_tol ** 2:
            if collision_free_segment(free, nx, ny, gx, gy):
                goal_node = Node(gx, gy, new_node)
                # Build path
                path = []
                cur = goal_node
                while cur is not None:
                    path.append((cur.x, cur.y))
                    cur = cur.parent
                path.reverse()
                return path, nodes

    return None, nodes



def generate_maze(width=640, height=480, cell=20, wall_thickness=3, seed=None):
    if seed is not None:
        random.seed(seed)

    cols = width // cell
    rows = height // cell
    cols = max(cols, 5)
    rows = max(rows, 5)

    visited = [[False] * cols for _ in range(rows)]
    walls = [[[True, True, True, True] for _ in range(cols)] for _ in range(rows)]

    def neighbors(r, c):
        out = []
        if r > 0 and not visited[r - 1][c]:
            out.append((r - 1, c, 0))  # top
        if c < cols - 1 and not visited[r][c + 1]:
            out.append((r, c + 1, 1))  # right
        if r < rows - 1 and not visited[r + 1][c]:
            out.append((r + 1, c, 2))  # bottom
        if c > 0 and not visited[r][c - 1]:
            out.append((r, c - 1, 3))  # left
        return out

    # DFS stack
    stack = [(0, 0)]
    visited[0][0] = True

    while stack:
        r, c = stack[-1]
        nb = neighbors(r, c)
        if not nb:
            stack.pop()
            continue
        nr, nc, direction = random.choice(nb)

        # knock down walls between (r,c) and (nr,nc)
        if direction == 0:  # top
            walls[r][c][0] = False
            walls[nr][nc][2] = False
        elif direction == 1:  # right
            walls[r][c][1] = False
            walls[nr][nc][3] = False
        elif direction == 2:  # bottom
            walls[r][c][2] = False
            walls[nr][nc][0] = False
        else:  # left
            walls[r][c][3] = False
            walls[nr][nc][1] = False

        visited[nr][nc] = True
        stack.append((nr, nc))

    # Rasterize walls into an image
    img = np.ones((rows * cell, cols * cell), dtype=np.uint8) * 255  # free = white

    # Draw walls
    for r in range(rows):
        for c in range(cols):
            x = c * cell
            y = r * cell
            t, rgt, b, lft = walls[r][c]
            if t:
                cv2.line(img, (x, y), (x + cell, y), 0, wall_thickness)
            if rgt:
                cv2.line(img, (x + cell, y), (x + cell, y + cell), 0, wall_thickness)
            if b:
                cv2.line(img, (x, y + cell), (x + cell, y + cell), 0, wall_thickness)
            if lft:
                cv2.line(img, (x, y), (x, y + cell), 0, wall_thickness)

    # Pad to requested size
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    return img


def find_free_point(free, prefer="top_left"):
    h, w = free.shape[:2]
    # search in region for a free pixel
    if prefer == "top_left":
        xs = range(5, w // 3)
        ys = range(5, h // 3)
    elif prefer == "bottom_right":
        xs = range(2 * w // 3, w - 5)
        ys = range(2 * h // 3, h - 5)
    else:
        xs = range(5, w - 5)
        ys = range(5, h - 5)

    for _ in range(20000):
        x = random.choice(list(xs))
        y = random.choice(list(ys))
        if is_free(free, x, y):
            return (x, y)
    # fallback brute
    ys2, xs2 = np.where(free > 0)
    if len(xs2) == 0:
        return (w // 2, h // 2)
    i = random.randint(0, len(xs2) - 1)
    return (int(xs2[i]), int(ys2[i]))


def draw_tree_and_path(maze_bgr, nodes, path):
    vis = maze_bgr.copy()

    # draw RRT edges
    for nd in nodes:
        if nd.parent is not None:
            cv2.line(vis, (nd.x, nd.y), (nd.parent.x, nd.parent.y), (180, 180, 180), 1)

    # draw path
    if path is not None and len(path) >= 2:
        for i in range(len(path) - 1):
            cv2.line(vis, path[i], path[i + 1], (255, 255, 0), 3)

    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maze", default=None, help="Path to maze image (white=free, black=wall). If omitted, generates a random maze.")
    ap.add_argument("--save_video", action="store_true", help="Save an mp4 of RRT growth.")
    ap.add_argument("--out_img", default="rrt_maze_result.png")
    ap.add_argument("--out_vid", default="rrt_maze_run.mp4")
    args = ap.parse_args()

    # Load or generate maze
    if args.maze is None:
        maze = generate_maze(width=900, height=600, cell=25, wall_thickness=4, seed=7)
    else:
        maze = cv2.imread(args.maze, cv2.IMREAD_GRAYSCALE)
        if maze is None:
            raise SystemExit(f"Could not read maze image: {args.maze}")

    # Ensure binary: white free, black walls
    _, maze_bin = cv2.threshold(maze, 127, 255, cv2.THRESH_BINARY)

    # Inflate walls slightly (makes planning realistic)
    wall = (maze_bin == 0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    wall = cv2.dilate(wall, k, iterations=1)
    free = cv2.bitwise_not(wall)

    h, w = free.shape[:2]
    start = (1,h//2)
    goal = ((w-10, h//6))

    # Run RRT (and optionally animate)
    maze_bgr = cv2.cvtColor(maze_bin, cv2.COLOR_GRAY2BGR)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = None
    if args.save_video:
        video = cv2.VideoWriter(args.out_vid, fourcc, 30, (w, h))

    # We want an animated version, so we re-implement RRT loop with drawing
    nodes = [Node(start[0], start[1], None)]
    path = None

    max_iter = 200000
    step = 18
    goal_bias = 0.18
    goal_tol = 12

    for it in range(max_iter):
        if random.random() < goal_bias:
            rx, ry = goal
        else:
            rx = random.randint(0, w - 1)
            ry = random.randint(0, h - 1)

        nn = nearest(nodes, rx, ry)
        nx, ny = steer(nn.x, nn.y, rx, ry, step)

        if not is_free(free, nx, ny):
            continue
        if not collision_free_segment(free, nn.x, nn.y, nx, ny):
            continue

        new_node = Node(nx, ny, nn)
        nodes.append(new_node)

        # draw growth occasionally
        if it % 30 == 0:
            vis = maze_bgr.copy()
            for nd in nodes:
                if nd.parent is not None:
                    cv2.line(vis, (nd.x, nd.y), (nd.parent.x, nd.parent.y), (180, 180, 180), 1)

            cv2.circle(vis, start, 6, (0, 255, 0), -1)
            cv2.circle(vis, goal, 6, (255, 0, 0), -1)
            cv2.putText(vis, f"RRT iterations: {it}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, f"RRT iterations: {it}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("RRT Maze Demo", vis)
            if video is not None:
                video.write(vis)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        # connect to goal
        if (nx - goal[0]) ** 2 + (ny - goal[1]) ** 2 <= goal_tol ** 2:
            if collision_free_segment(free, nx, ny, goal[0], goal[1]):
                goal_node = Node(goal[0], goal[1], new_node)
                path = []
                cur = goal_node
                while cur is not None:
                    path.append((cur.x, cur.y))
                    cur = cur.parent
                path.reverse()
                break

    # Final visualization
    final_vis = draw_tree_and_path(maze_bgr, nodes, path)
    cv2.circle(final_vis, start, 6, (0, 255, 0), -1)
    cv2.circle(final_vis, goal, 6, (255, 0, 0), -1)

    if path is None:
        cv2.putText(final_vis, "NO PATH FOUND", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(final_vis, "NO PATH FOUND", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(final_vis, f"PATH FOUND (nodes={len(nodes)})", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(final_vis, f"PATH FOUND (nodes={len(nodes)})", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imwrite(args.out_img, final_vis)
    print("Saved:", args.out_img)
    if video is not None:
        # write a few final frames
        for _ in range(30):
            video.write(final_vis)
        video.release()
        print("Saved:", args.out_vid)

    cv2.imshow("RRT Maze Demo", final_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
