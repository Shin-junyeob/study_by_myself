import sys
n, d = map(int, sys.stdin.readline().split())
lst = [0 for _ in range(n)]
for i in range(n):
    lst[i] = list(map(int, sys.stdin.readline().rstrip().split()))
dis = [_ for _ in range(d+1)]

for i in range(d+1):
    if i > 0:
        dis[i] = min(dis[i], dis[i-1]+1)
    for a, b, l in lst:
        if i == a and b <= d and dis[i]+l < dis[b]:
            dis[b] = dis[i]+l
print(dis[d])