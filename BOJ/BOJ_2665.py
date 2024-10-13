import heapq
import sys
n = int(sys.stdin.readline())
lst = [0 for _ in range(n)]
INF = int(1e4)
check = [[False for _ in range(n)] for _ in range(n)]
dis = [[INF for _ in range(n)] for _ in range(n)]
for i in range(n):
    lst[i] = list(map(int, sys.stdin.readline().rstrip()))

dx = [-1, 1, 0 ,0]
dy = [0, 0, -1, 1]

def dijkstra(start_x, start_y):
    heap = []
    heapq.heappush(heap, (0, start_x, start_y))
    check[start_x][start_y] = True
    dis[start_x][start_y] = 0
    while heap:
        d, x, y = heapq.heappop(heap)
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            if nx < 0 or nx >= n or ny < 0 or ny >= n:
                continue
            if lst[nx][ny] == 0 and check[nx][ny] == False:
                check[nx][ny] = True
                dis[nx][ny] = d+1
                heapq.heappush(heap, (d+1, nx, ny))
            if lst[nx][ny] == 1 and check[nx][ny] == False:
                check[nx][ny] = True
                dis[nx][ny] = d
                heapq.heappush(heap, (d, nx, ny))
dijkstra(0, 0)

print(dis[-1][-1])