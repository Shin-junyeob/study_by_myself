import sys
from collections import deque
n = int(sys.stdin.readline())
lst = []
for _ in range(n):
    lst.append(list(map(int, sys.stdin.readline().rstrip().split())))

dx = [-1, 1, 0, 0]; dy = [0, 0, -1, 1]
visited = [[0 for _ in range(n)] for _ in range(n)]

def bfs(x, y):
    queue = deque()
    queue.append((x, y))
    visited[x][y] = 1
    while queue:
        x, y = queue.popleft()
        if x == n-1 and y == n-1:
            return 'HaruHaru'
        dis = lst[x][y]
        if x == n-1 and y == n-1:
            answer = 'HaruHaru'
        for i in range(4):
            nx = x + dx[i] * dis
            ny = y + dy[i] * dis
            if nx < 0 or ny < 0 or nx >= n or ny >= n:
                continue
            if visited[nx][ny] != 0:
                continue
            if visited[nx][ny] == 0:
                visited[nx][ny] = 1
                queue.append((nx ,ny))
    return 'Hing'

print(bfs(0, 0))