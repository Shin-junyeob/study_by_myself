import sys

n = int(sys.stdin.readline())
lst = [0 for x in range(n)]
lst2 = [0 for x in range(n)]

for i in range(n):
    lst[i] = int(sys.stdin.readline().rstrip())

lst.sort(reverse = True)

for i in range(n):
    lst2[i] = (i+1)*lst[i]

print(max(lst2))