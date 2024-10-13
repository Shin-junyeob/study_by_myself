import sys
from collections import deque

f, S, G, U, D = map(int, sys.stdin.readline().split())
if S > G:
    s = G; g = S; u = D; d = U
else:
    s = S; g = G; u = U; d = D
visited = [-1 for x in range(f+u+1)]

q = deque()
q.append(s)
visited[s] = 0
while q:
    now = q.popleft()
    for next in (now + u, now - d):
        if 0 < next <= f and visited[next] == -1:
            visited[next] = visited[now] + 1
            q.append(next)

if visited[g] == -1:
    print('use the stairs')
else:
    print(visited[g])