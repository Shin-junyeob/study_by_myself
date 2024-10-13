import sys, heapq

n, m, r = map(int, sys.stdin.readline().split())
items = list(map(int, sys.stdin.readline().split()))
graph = [[] for _ in range(n+1)]
for _ in range(r):
    a, b, l = map(int, sys.stdin.readline().rstrip().split())
    graph[a].append((b, l))
    graph[b].append((a, l))

INF = int(1e9)
answer = 0

def dijkstra(start):
    heap = []
    heapq.heappush(heap, (0, start))
    distance[start] = 0
    while heap:
        dist, now = heapq.heappop(heap)
        if distance[now] < dist:
            continue
        for i in graph[now]:
            cost = dist + i[1]
            if cost < distance[i[0]]:
                distance[i[0]] = cost
                heapq.heappush(heap, (cost, i[0]))

for i in range(1, n+1):
    distance = [INF] * (n+1)
    dijkstra(i)
    result = 0
    for j in range(1, n+1):
        if distance[j] <= m:
            result += items[j-1]
    answer = max(answer, result)

print(answer)