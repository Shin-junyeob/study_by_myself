import sys
from collections import defaultdict, deque

n = int(sys.stdin.readline())
dic = defaultdict(list)
lst = [int(x) for x in sys.stdin.readline().split()]
k = int(sys.stdin.readline())
for l, node in enumerate(lst):
    if  node != k and l != k:
        dic[node].append(l)

print(dic)
leaf = 0

def func(dic, node):
    global leaf
    queue = deque()
    if node in dic:
        queue.append(node)
    while queue:
        node = queue.popleft()
        if node in dic:
            for i in dic[node]:
                queue.append(i)
        else:
            leaf += 1
    return leaf

print(func(dic, -1))