import sys
n, m = map(int, sys.stdin.readline().split())
lst1 = [int(x) for x in sys.stdin.readline().split()]
lst2 = [int(x) for x in sys.stdin.readline().split()]

lst = sorted(lst1 + lst2)

for i in lst:
    print(i, end = ' ')
