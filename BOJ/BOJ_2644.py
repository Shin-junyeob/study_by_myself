import sys
from collections import defaultdict
from collections import deque

n = int(sys.stdin.readline())
x, y = map(int, sys.stdin.readline().split())
m = int(sys.stdin.readline())
dic = defaultdict(list)
for _ in range(m):
    a, b = map(int, sys.stdin.readline().split())
    dic[a].append(b)
    dic[b].append(a)

def bfs(dic, x, y, visited):
    queue = deque()
    queue.append(x)
    visited[x] = 0
    while queue:
        x = queue.popleft()
        for i in dic[x]:
            if visited[i] == -1:
                queue.append(i)
                visited[i] = visited[x] + 1
    return visited[y]

visited = [-1 for _ in range(n+1)]

print(bfs(dic, x, y, visited))