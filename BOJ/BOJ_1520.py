import sys
sys.setrecursionlimit(100000000)
m, n = map(int, sys.stdin.readline().split())
graph = []
for _ in range(m):
    graph.append(list(map(int, sys.stdin.readline().rstrip().split())))

dx = [-1, 1, 0 ,0]; dy = [0, 0, -1, 1]
visited = [[-1 for _ in range(n)] for _ in range(m)]

def dfs(x, y):
    global answer
    if x == m-1 and y == n-1:
        return 1
    if visited[x][y] == -1:
        visited[x][y] = 0
        for i in range(4):
            nx = x + dx[i]; ny = y + dy[i]
            if 0 <= nx < m and 0 <= ny < n and graph[x][y] > graph[nx][ny]:
                visited[x][y] += dfs(nx, ny)
    return visited[x][y]

print(dfs(0, 0))