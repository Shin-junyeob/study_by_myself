import sys
n = int(sys.stdin.readline())
lst = [0 for x in range(n)]
for i in range(n):
    lst[i] = sys.stdin.readline().rstrip()
lst = list(set(lst))

def func1(arr):
    if len(arr) <= 1:
        return arr
    p = arr[len(arr) // 2]
    left, middle, right = [], [], []
    for i in range(len(arr)):
        if len(arr[i]) < len(p):
            left.append(arr[i])
        elif len(arr[i]) > len(p):
            right.append(arr[i])
        else:
            middle.append(arr[i])
    return func1(sorted(left)) + sorted(middle) + func1(sorted(right))

ans = func1(lst)
for i in ans:
    print(i)