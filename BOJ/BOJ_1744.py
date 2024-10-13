import sys
n = int(sys.stdin.readline())
lst1 = []
lst2 = []
lst3 = []
for i in range(n):
    num = int(sys.stdin.readline().rstrip())
    if num > 0:
        lst1.append(num)
    elif num < 0:
        lst2.append(num)
    else:
        lst3.append(num)

lst1.sort(), lst2.sort(), lst3.sort()

def func():
    cnt = 0
    while len(lst1) > 1:
        a = lst1.pop()
        b = lst1.pop()
        cnt += max(a*b, a+b)
    while len(lst2) > 1:
        a = lst2.pop(0)
        b = lst2.pop(0)
        cnt += a*b
    if lst1:
        cnt += lst1[0]
    if lst2:
        if lst3:
            pass
        if not lst3:
            cnt += lst2[0]
    return cnt
print(func())