import sys
from collections import deque, defaultdict

n = int(sys.stdin.readline())
graph = defaultdict(list)
for i in range(1, n+1):
    num = int(sys.stdin.readline().rstrip())
    graph[i] = num

def dfs(start):
    queue = deque()
    queue.append(start)
    visited[start] = 1
    check = [start]
    while queue:
        node = queue.popleft()
        nnode = graph[node]
        if nnode not in check:
            check.append(nnode)
            queue.append(nnode)
        else:
            if nnode == start:
                answer.extend(check)
            else:
                break


answer = []
for i in range(1, n+1):
    visited = [0 for _ in range(n+1)]
    dfs(i)
answer = sorted(list(set(answer)))
print(len(answer))
for i in answer:
    print(i)