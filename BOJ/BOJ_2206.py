import sys
from collections import deque
n, m = map(int, sys.stdin.readline().split())
lst = []
for _ in range(n):
    lst.append(list(map(int, sys.stdin.readline().rstrip())))

visited = [[[0]*2 for _ in range(m)] for _ in range(n)]
dx, dy = [-1, 1, 0, 0], [0, 0, -1, 1]
def bfs(x, y, c):
    queue = deque()
    queue.append((x, y, c))
    visited[x][y][c] = 1
    while queue:
        x, y, c = queue.popleft()
        if x == n-1 and y == m-1:
            return visited[x][y][c]
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            if nx < 0 or ny < 0 or nx >= n or ny >= m:
                continue
            if lst[nx][ny] == 1 and c == 0:
                visited[nx][ny][1] = visited[x][y][0] + 1
                queue.append((nx, ny, 1))
            elif lst[nx][ny] == 0 and visited[nx][ny][c] == 0:
                visited[nx][ny][c] = visited[x][y][c] + 1
                queue.append((nx, ny, c))
    return -1
print(bfs(0, 0, 0))