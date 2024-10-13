import sys
from collections import deque

dx = [-1, 1, 0, 0, 1, 1, -1, -1]; dy = [0, 0, -1, 1, 1, -1, 1, -1]
def bfs(x, y):
    global count
    global w
    global h
    queue = deque()
    queue.append((x, y))
    visited[x][y] = count
    while queue:
        x, y = queue.popleft()
        for i in range(8):
            nx = x + dx[i]; ny = y + dy[i]
            if 0 <= nx < h and 0 <= ny < w and visited[nx][ny] == 0 and graph[nx][ny] == 1:
                visited[nx][ny] = count
                queue.append((nx, ny))

while True:
    w, h = map(int, sys.stdin.readline().rstrip().split())
    if w == h == 0:
        break
    graph = []
    for _ in range(h):
        graph.append(list(map(int, sys.stdin.readline().rstrip().split())))
    visited = [[0 for _ in range(w)] for _ in range(h)]
    count = 0
    for i in range(h):
        for j in range(w):
            if visited[i][j] == 0 and graph[i][j] == 1:
                count += 1
                bfs(i, j)
    answer = 0
    for i in visited:
        answer = max(answer, max(i))
    print(answer)