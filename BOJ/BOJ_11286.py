import sys, heapq
n = int(sys.stdin.readline())
heap = []
for i in range(n):
    num = int(sys.stdin.readline().rstrip())
    if num == 0:
        try:
            a, b = heapq.heappop(heap)
            print(b)
        except:
            print(0)
    else:
        heapq.heappush(heap, (abs(num), num))