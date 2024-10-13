import sys
from collections import deque

n, m = map(int, sys.stdin.readline().split())
lst = [0 for x in range(n)]
for i in range(n):
    lst[i] = list(map(int, sys.stdin.readline().rstrip()))

def bfs(x, y):
    queue = deque()
    queue.append((x, y))
    while queue:
        x, y = queue.popleft()
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            if nx < 0 or nx >= n or ny < 0 or ny >= m:
                continue
            if lst[nx][ny] == 0:
                continue
            if lst[nx][ny] == 1:
                lst[nx][ny] = lst[x][y] + 1
                queue.append((nx, ny))
    return lst[n-1][m-1]

dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]

print(bfs(0, 0))