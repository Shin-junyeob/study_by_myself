import sys
from collections import deque

n = int(sys.stdin.readline())
matrix = [0 for x in range(n)]
for i in range(n):
    matrix[i] = list(map(int, sys.stdin.readline().rstrip()))

count = []
def bfs(matrix, x, y, visited):
    cnt = 0
    queue = deque()
    queue.append((x, y))
    while queue:
        x, y = queue.popleft()
        if matrix[x][y] == 1 and visited[x][y] == 0:
            visited[x][y] = 1
            cnt += 1
            for i in range(4):
                nx = x + dx[i]
                ny = y + dy[i]
                if nx < 0 or nx >= n or ny < 0 or ny >= n:
                    continue
                if matrix[nx][ny] == 0:
                    continue
                if matrix[nx][ny] == 1 and visited[nx][ny] == 0:
                    queue.append((nx, ny))
    if cnt > 0:
        count.append(cnt)
    return count

visited = [[0 for _ in range(n)] for _ in range(n)]
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]

# print('\n# # # # # # # # # # # # # #\n')
# ans = bfs(matrix, 0, 1, visited)
# print(ans)

# for i in visited:
#     print(i)

for i in range(n):
    for j in range(n):
        ans = bfs(matrix, i, j, visited)

print(len(ans))
for i in sorted(ans):
    print(i)