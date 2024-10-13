import sys, re
from collections import defaultdict
n = int(sys.stdin.readline())
dic = defaultdict(list)
for _ in range(n):
  a, b = sys.stdin.readline().rstrip().split('.')
  dic[b].append(a)

lst = []
for i in dic:
  lst.append([i, len(dic[i])])
lst.sort()
for i in lst:
  for j in i:
    print(j ,end = ' ')
  print()