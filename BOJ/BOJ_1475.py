import sys
a = sys.stdin.readline()

lst = [0 for x in range(len(a))]
for i in range(len(a)):
    lst[i] = a[i]

lst.pop()

lst2 = [0 for x in range(9)]
for i in range(len(lst)):
    num = int(lst[i])
    if num != 9:
        lst2[num] += 1
    else:
        lst2[6] += 1

if lst2[6] % 2 != 0:
    lst2[6] += 1

lst2[6] //= 2

print(max(lst2))