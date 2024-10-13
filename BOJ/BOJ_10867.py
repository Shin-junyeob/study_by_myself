import sys

n = int(sys.stdin.readline())
lst = [int(x) for x in sys.stdin.readline().split()]
lst = sorted(list(set(lst)))

for i in lst:
    print(i, end = ' ')