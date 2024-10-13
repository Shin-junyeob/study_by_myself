import sys
sys.setrecursionlimit(100000000)
n = int(sys.stdin.readline())
graph = [[] for _ in range(n+1)]
for _ in range(n-1):
    a, b, c = map(int, sys.stdin.readline().rstrip().split())
    graph[a].append([b, c])
    graph[b].append([a, c])

distance = [-1 for _ in range(n+1)]
distance[1] = 0
def dfs(node, dis):
    for i in graph[node]:
        nnode, ndis = i
        if distance[nnode] == -1:
            distance[nnode] = dis + ndis
            dfs(nnode, dis+ndis)

dfs(1, 0)
start_node, start_dis = distance.index(max(distance)), max(distance)
distance = [-1 for _ in range(n+1)]
distance[start_node] = 0
dfs(start_node, 0)
print(max(distance))