'''
입력 :
5
3 1 4 3 2
'''

import sys
import time
from bigO import BigO
from bigO import algorithm

lib = BigO()

start_time = time.time()
n = int(sys.stdin.readline())
lst = [int(x) for x in (sys.stdin.readline().split())]

def func(lst):
    if len(lst) <= 1:
        return lst
    pivot = lst[len(lst) // 2]
    left, middle, right = [], [], []
    for i in range(len(lst)):
        if lst[i] < pivot:
            left.append(lst[i])
        elif lst[i] > pivot:
            right.append(lst[i])
        else:
            middle.append(lst[i])
    return func(left) + middle + func(right)

ans = func(lst)

for i in range(len(ans)):
    if i == 0:
        ans[i] = ans[i]
    else:
        ans[i] = ans[i] + ans[i-1]

print(ans)
print(sum(ans))

lib.test(func, 'random')
end_time = time.time()
print(end_time - start_time)