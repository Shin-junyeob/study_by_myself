import sys
from collections import deque

t = int(sys.stdin.readline())
for _ in range(t):
    n = int(sys.stdin.readline().rstrip())
    graph = [0 for _ in range(n+1)]
    for _ in range(n-1):
        a, b = map(int, sys.stdin.readline().rstrip().split())
        graph[b] = a
    x, y = map(int, sys.stdin.readline().rstrip().split())
    check = [x, y]
    queue = deque()
    queue.append(x)
    queue.append(y)
    while queue:
        node = queue.popleft()
        if graph[node] == 0:
            pass
        else:
            if graph[node] in check:
                print(graph[node])
                break
            else:
                check.append(graph[node])
                queue.appendleft(graph[node])