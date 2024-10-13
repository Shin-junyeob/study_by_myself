import sys
from collections import deque

n = int(sys.stdin.readline())
graph = []
for _ in range(n):
    graph.append(list(map(str, sys.stdin.readline().rstrip())))

visited = [[0 for _ in range(n)] for _ in range(n)]
dx = [-1, 1, 0 ,0]; dy = [0, 0, -1, 1]

dcount = 0
def different(x, y):
    queue = deque()
    queue.append((x, y))
    visited[x][y] = 1
    while queue:
        x, y = queue.popleft()
        for i in range(4):
            nx = x + dx[i]; ny = y + dy[i]
            if 0 <= nx < n and 0 <= ny < n and visited[nx][ny] == 0 and graph[x][y] == graph[nx][ny]:
                visited[nx][ny] = 1
                queue.appendleft((nx, ny))

scount = 0
def same(x, y):
    queue = deque()
    queue.append((x, y))
    visited[x][y] = 2
    while queue:
        x, y = queue.popleft()
        for i in range(4):
            nx = x + dx[i]; ny = y + dy[i]
            if 0 <= nx < n and 0 <= ny < n and visited[nx][ny] == 1:
                if graph[x][y] == 'R' or graph[x][y] == 'G':
                    if graph[nx][ny] == 'R' or graph[nx][ny]== 'G':
                        visited[nx][ny] = 2
                        queue.appendleft((nx, ny))
                else:
                    if graph[nx][ny] == 'B':
                        visited[nx][ny] = 2
                        queue.appendleft((nx, ny))

for i in range(n):
    for j in range(n):
        if visited[i][j] == 0:
            dcount += 1
            different(i, j)

for i in range(n):
    for j in range(n):
        if visited[i][j] == 1:
            scount += 1
            same(i, j)

print(dcount, scount)