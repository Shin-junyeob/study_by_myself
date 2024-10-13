import sys
n = int(sys.stdin.readline())
lst = [0 for x in range(n)]
for i in range(n):
    lst[i] = list(map(int, sys.stdin.readline().split()))

lst2 = [1 for x in range(n)]
for i in range(n):
    for j in range(n):
        if (lst[i][0] < lst[j][0]) and (lst[i][1] < lst[j][1]):
            lst2[i] += 1

for i in lst2:
    print(i, end = ' ')