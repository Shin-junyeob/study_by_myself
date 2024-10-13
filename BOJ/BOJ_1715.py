import sys, heapq
n = int(sys.stdin.readline())
heap = []
for i in range(n):
    num = int(sys.stdin.readline().rstrip())
    heapq.heappush(heap, num)

cnt = 0
while heap:
    if len(heap) <= 1:
        print(cnt)
        exit()
    else:
        card1 = heapq.heappop(heap)
        card2 = heapq.heappop(heap)
        cnt += card1 + card2
        heapq.heappush(heap, card1 + card2)