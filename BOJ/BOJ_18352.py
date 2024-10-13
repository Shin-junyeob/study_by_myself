import sys
from collections import defaultdict
from collections import deque

n, m, k, x = map(int, sys.stdin.readline().split())
dic = defaultdict(list)
for _ in range(m):
    a, b = map(int, sys.stdin.readline().split())
    dic[a].append(b)

ans = []
def bfs(dic, x, k, visited):
    queue = deque()
    queue.append(x)
    visited[x] = 0
    while queue:
        x = queue.popleft()
        for i in dic[x]:
            if visited[i] == -1:
                queue.append(i)
                visited[i] = visited[x] + 1
        if visited[x] == k:
            break
    for i in range(len(visited)):
        if visited[i] == k:
            ans.append(i)
    return sorted(ans)

visited = [-1 for _ in range(n+1)]

result = bfs(dic, x, k, visited)
if len(result) == 0:
    print(-1)
else:
    for i in result:
        print(i)