import sys
n = int(sys.stdin.readline())
lst = [int(x) for x in sys.stdin.readline().split()]

count = [1 for x in range(n)]
for i in range(1, n):
    for j in range(i):
        if lst[i] < lst[j]:
            count[i] = max(count[i], count[j]+1)


print(n - max(count))