<<<<<<< HEAD
import sys, heapq
n = int(sys.stdin.readline())
heap = []
for i in range(n):
    num = int(sys.stdin.readline().rstrip())
    if num == 0:
        try:
            print(heapq.heappop(heap))
        except:
            print(0)
    else:
=======
import sys, heapq
n = int(sys.stdin.readline())
heap = []
for i in range(n):
    num = int(sys.stdin.readline().rstrip())
    if num == 0:
        try:
            print(heapq.heappop(heap))
        except:
            print(0)
    else:
>>>>>>> 77e016886 (Initial commit)
        heapq.heappush(heap, num)