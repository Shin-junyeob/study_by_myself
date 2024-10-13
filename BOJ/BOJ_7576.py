import sys
from collections import deque

m, n = map(int, sys.stdin.readline().split())
tomato_box = []
for _ in range(n):
    tomato_box.append(list(map(int, sys.stdin.readline().rstrip().split())))

tomato = []
visited = [[0 for _ in range(m)] for _ in range(n)]
for i in range(n):
    for j in range(m):
        if tomato_box[i][j] == 1:
          tomato.append((i, j))
        if tomato_box[i][j] == -1:
            visited[i][j] = -1

dx = [-1, 1, 0, 0]; dy = [0, 0, -1, 1]
time = 0
queue = deque()
for i in tomato:
    queue.append(i)
    visited[i[0]][i[1]] = 1
while queue:
    x, y = queue.popleft()
    for i in range(4):
        nx = x + dx[i]; ny = y + dy[i]
        if nx < 0 or ny < 0 or nx >= n or ny >= m:
            continue
        if tomato_box[nx][ny] == 0 and visited[nx][ny] == 0:
            visited[nx][ny] = visited[x][y] + 1
            queue.append((nx, ny))

answer = 0
for i in visited:
    if 0 in i:
        answer = -1
        break
    else:
        answer = max(answer, max(i)-1)
print(answer)