import sys
from collections import deque

n = int(sys.stdin.readline())
graph = [list(map(int, sys.stdin.readline().rstrip().split())) for _ in range(n)]
hmax = max(map(max, graph)); hmin = min(map(min, graph))

def bfs(x, y, h):
    queue = deque()
    queue.append((x, y))
    visited[x][y] = 1
    while queue:
        x, y = queue.popleft()
        for i in range(4):
            nx = x + dx[i]; ny = y + dy[i]
            if 0 <= nx < n and 0 <= ny < n and visited[nx][ny] == 0 and graph[nx][ny] >= h:
                visited[nx][ny] = 1
                queue.append((nx, ny))

answer = 0
dx = [-1, 1, 0, 0]; dy = [0, 0, -1, 1]
for mmax in range(hmin, hmax+1):
    visited = [[0 for _ in range(n)] for _ in range(n)]
    count = 0

    for i in range(n):
        for j in range(n):
            if visited[i][j] == 0 and graph[i][j] >= mmax:
                count += 1
                bfs(i, j, mmax)
    # print(f'높이: {mmax}')
    # for i in visited:
    #     print(i)
    # print(f'섬의 갯수: {count}')
    # print()
    answer = max(answer, count)

print(answer)