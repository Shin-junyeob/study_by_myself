import heapq, sys
from collections import defaultdict
n, m = map(int, sys.stdin.readline().split())
dic = defaultdict(list)
for i in range(m):
    s, e = map(int, sys.stdin.readline().rstrip().split())
    dic[s].append(e)
    dic[e].append(s)
INF = int(1e9)
distance = [INF for _ in range(n)]
check = [False for _ in range(n)]
def dijkstra(start):
    heap = []
    heapq.heappush(heap, (0, start))
    distance[start-1] = 0
    while heap:
        if not check:
            break
        dis, now = heapq.heappop(heap)
        for i in dic[now]:
            if check[i-1] == False:
                distance[i-1] = min(distance[i-1], dis+1)
                heapq.heappush(heap, (dis+1, i))
                check[i-1] = True
dijkstra(1)

id = distance.index(max(distance))+1
d = distance[id-1]
cnt = 0
for i in distance:
    if i == d:
        cnt += 1
print(id, d, cnt)