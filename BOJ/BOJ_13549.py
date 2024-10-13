import sys
from collections import deque

n, k = map(int, sys.stdin.readline().split())
q = deque()
q.append(n)
check = [-1 for _ in range(100001)]
check[n] = 0

while q:
    now = q.popleft()
    if now == k:
        print(check[now])
        exit()
    if 0 <= now-1 <100001 and check[now-1] == -1:
        check[now-1] = check[now] + 1
        q.append(now-1)
    if 0 <= now*2 < 100001 and check[now*2] == -1:
        check[now*2] = check[now]
        q.append(now*2)
    if 0 <= now+1 < 100001 and check[now+1] == -1:
        check[now+1] = check[now] + 1
        q.append(now+1)