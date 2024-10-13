import sys
n = int(sys.stdin.readline().rstrip())
lst = []
for i in range(n):
    lst.append(int(sys.stdin.readline().rstrip()))

lst.sort()
for i in range(n):
    print(lst[i])