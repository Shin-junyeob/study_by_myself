import sys
from collections import defaultdict
sys.setrecursionlimit(1000000000)
n = int(sys.stdin.readline())
dic = defaultdict(list)
for i in range(n-1):
    a, b = map(int, sys.stdin.readline().rstrip().split())
    dic[a].append(b)
    dic[b].append(a)

visited = [False for _ in range(n+1)]
res = defaultdict(list)
def dfs(dic, node):
    visited[node] = True
    for i in dic[node]:
        if visited[i] == False:
            res[node].append(i)
            dfs(dic, i)

dfs(dic, 1)

ans = [0 for _ in range(n+1)]
for i in res:
    for j in res[i]:
        ans[j] = i

for i in range(2, n+1):
    print(ans[i])