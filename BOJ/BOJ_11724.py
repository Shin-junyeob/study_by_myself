import sys
from collections import deque
from collections import defaultdict
sys.setrecursionlimit(10**9)

n, m = map(int, sys.stdin.readline().split())
matrix = [[0 for _ in range(n+1)] for _ in range(n+1)]
dic = defaultdict(list)

for i in range(m):
    u, v = map(int, sys.stdin.readline().rstrip().split())
    matrix[u][v] = matrix[v][u] = 1
    dic[u].append(v)
    dic[v].append(u)

for i in matrix:
    print(i)

# DFS 풀이
answer = []
def dfs(matrix, x, visited):
    ans = []
    if visited[x] == 0:
        visited[x] = 1
        ans.append(x)
    for i in range(1, n+1):
        if matrix[x][i] == 1 and visited[i] == 0:
            visited[i] = 1
            dfs(matrix, i, visited)
    if len(ans) > 0 and ans not in answer:
        answer.append(ans)
    return answer

visited_dfs = [0 for x in range(n+1)]

for i in range(1,n+1):
    result = dfs(matrix, i, visited_dfs)
print(f'DFS: {len(result)}')

# BFS 풀이
answer = []
ans = []
def bfs(dic, x, visited):
    global ans
    queue = deque()
    queue.append(x)
    if visited[x] == 0:
        visited[x] = 1
        ans.append(x)
    while queue:
        x = queue.popleft()
        for i in dic[x]:
            if visited[i] == 0:
                visited[i] = 1
                queue.append(i)
                ans.append(i)
    answer.append(ans)
    ans = []
    return answer

visited_bfs = [0 for _ in range(n+1)]
for i in range(1, n+1):
    res = bfs(dic, i, visited_bfs)

result = []
for i in res:
    if len(i) > 0:
        result.append(i)
print(f'BFS: {len(result)}')