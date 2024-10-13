import sys
from collections import deque

d = [[1, 2], [2, 1], [2, -1], [1, -2], [-1, -2], [-2, -1], [-2, 1], [-1, 2]]
def bfs(start_x, start_y):
    queue = deque()
    queue.append((start_x, start_y))
    while queue:
        x, y = queue.popleft()
        if x == ex and y == ey:
            return visited[x][y]
        for i in range(8):
            nx, ny = x+d[i][0], y+d[i][1]
            if nx < 0 or ny < 0 or nx >= l or ny >= l:
                continue
            if visited[nx][ny] != 0:
                continue
            if visited[nx][ny] == 0:
                visited[nx][ny] = visited[x][y] + 1
                queue.append((nx, ny))
    return 0

t = int(sys.stdin.readline())
for _ in range(t):
    l = int(sys.stdin.readline().rstrip())
    visited = [[0 for _ in range(l)] for _ in range(l)]
    sx, sy = map(int, sys.stdin.readline().split())
    ex, ey = map(int, sys.stdin.readline().split())
    print(bfs(sx, sy))