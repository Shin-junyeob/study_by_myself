import sys
from itertools import permutations
n, m = map(int, sys.stdin.readline().split())
lst = [int(x) for x in range(1,n+1)]
res = list(permutations(lst, m))
for i in res:
    for j in i:
        print(j, end=' ')
    print()