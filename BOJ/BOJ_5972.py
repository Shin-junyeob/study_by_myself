import sys, heapq
from collections import defaultdict
n, m = map(int, sys.stdin.readline().split())
dic = defaultdict(list)
for i in range(m):
    a, b, c = map(int, sys.stdin.readline().rstrip().split())
    dic[a].append((b, c))
    dic[b].append((a, c))

INF = int(1e9)
distance = [INF] * (n+1)
def dijkstra(start):
    heap = []
    heapq.heappush(heap, (0, start))
    distance[start] = 0
    while heap:
        cost, now = heapq.heappop(heap)
        for i in dic[now]:
            ncost = cost + i[1]
            if ncost < distance[i[0]]:
                distance[i[0]] = ncost
                heapq.heappush(heap, (ncost, i[0]))
dijkstra(1)
print(distance[n])