import sys
from collections import defaultdict
from collections import deque

n = int(sys.stdin.readline())
m = int(sys.stdin.readline())
dic = defaultdict(list)
for i in range(m):
    a, b = map(int, sys.stdin.readline().split())
    dic[a].append(b)
    dic[b].append(a)

# dfs 코드
def dfs(dic, s, visited):
    visited[s] = 1
    for i in dic[s]:
        if not visited[i]:
            visited[i] = 1
            dfs(dic, i, visited)
    return sum(visited)-1

# bfs 코드
def bfs(dic, s, visited):
    queue = deque()
    queue.append((s))
    visited[s] = 1
    while queue:
        s = queue.popleft()
        for i in dic[s]:
            if not visited[i]:
                visited[i] = 1
                queue.append(i)
    return sum(visited)-1

visited_dfs = [0 for x in range(n+1)]
visited_bfs = [0 for x in range(n+1)]
ans_dfs = dfs(dic, 1, visited_dfs)
ans_bfs = bfs(dic, 1, visited_bfs)

print(f'dfs.ver : {ans_dfs}')
print(f'bfs.ver : {ans_bfs}')