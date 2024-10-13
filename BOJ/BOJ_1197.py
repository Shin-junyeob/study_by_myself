import sys, heapq

v, e = map(int, sys.stdin.readline().split())
graph = [[] for _ in range(v+1)]
for _ in range(e):
    a, b, c = map(int, sys.stdin.readline().rstrip().split())
    graph[a].append([c, b])
    graph[b].append([c, a])

visited = [1] + [0] * v
distance = 0
cnt = 0
heap = []
heapq.heappush(heap, (0, 1))
while heap:
    if cnt == v:
        break
    dis, node = heapq.heappop(heap)
    if visited[node] == 0:
        visited[node] = 1
        distance += dis
        cnt += 1
        for i in graph[node]:
            heapq.heappush(heap, i)
print(distance)