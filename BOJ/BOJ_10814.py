import sys
n = int(sys.stdin.readline())
lst = [0 for _ in range(n)]

for i in range(n):
    lst[i] = list(sys.stdin.readline().split())

for i in range(n):
    lst[i][0] = int(lst[i][0])

lst = sorted(lst, key = lambda x : x[0])
for i in range(n):
    print(lst[i][0], lst[i][1])