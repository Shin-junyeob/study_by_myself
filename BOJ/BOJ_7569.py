import sys
from collections import deque

m, n, h = map(int, sys.stdin.readline().split())
graph = [[] for _ in range(h)]
tomato = []
not_tomato = []
INF = int(1e9)
visited = [[[INF for _ in range(m)] for _ in range(n)] for _ in range(h)]

for i in range(h):
    for j in range(n):
        lst = list(map(int, sys.stdin.readline().rstrip().split()))
        graph[i].append(lst)

for i in range(h):
    for j in range(n):
        for k in range(m):
            if graph[i][j][k] == 1:
                tomato.append((i, j, k))
            if graph[i][j][k] == -1:
                not_tomato.append((i, j, k))

for i in not_tomato:
    visited[i[0]][i[1]][i[2]] = -1
for i in tomato:
    visited[i[0]][i[1]][i[2]] = 1
dx = [-1, 1, 0, 0]; dy = [0, 0, -1, 1]
def bfs():
    queue = deque()
    for i in tomato:
        queue.append(i)
    while queue:
        r, x, y = queue.popleft()
        for i in range(4):
            nx = x + dx[i]; ny = y + dy[i]
            if 0 <= nx < n and 0 <= ny < m and graph[r][nx][ny] == 0 and visited[r][nx][ny] == INF:
                visited[r][nx][ny] = min(visited[r][nx][ny], visited[r][x][y] + 1)
                queue.append((r, nx, ny))
        if 0 <= r-1 < h and graph[r-1][x][y] == 0 and visited[r-1][x][y] == INF:
            visited[r-1][x][y] = min(visited[r-1][x][y], visited[r][x][y] + 1)
            queue.append((r-1, x, y))
        if 0 <= r+1 < h and graph[r+1][x][y] == 0 and visited[r+1][x][y] == INF:
            visited[r+1][x][y] = min(visited[r+1][x][y], visited[r][x][y] + 1)
            queue.append((r+1, x, y))

bfs()
mx = 0
for i in visited:
    for j in i:
        if INF in j:
            print(-1)
            exit()
        else:
            mx = max(mx, max(j))

print(mx-1)