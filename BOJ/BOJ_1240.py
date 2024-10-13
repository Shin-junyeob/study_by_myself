import sys
from collections import deque, defaultdict

n, m = map(int, sys.stdin.readline().split())
dic = defaultdict(list)
for _ in range(n-1):
    a, b, k = map(int, sys.stdin.readline().split())
    dic[a].append([b, k])
    dic[b].append([a, k])

def bfs(start, end):
    queue = deque()
    queue.append(start)
    visited = [False for _ in range(n+1)]
    visited[start] = True
    dis = [0 for _ in range(n+1)]
    while queue:
        start = queue.popleft()
        if start == end:
            print(dis[end])
        for i, l in dic[start]:
            if not visited[i]:
                queue.append(i)
                visited[i] = True
                dis[i] = dis[start] + l

for _ in range(m):
    start, end = map(int, sys.stdin.readline().split())
    bfs(start, end)