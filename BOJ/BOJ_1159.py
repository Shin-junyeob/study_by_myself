import sys
n = int(sys.stdin.readline())
lst = [0 for _ in range(n)]
for i in range(n):
    lst[i] = sys.stdin.readline().rstrip()[0]

check = sorted(list(set(lst)))
result = []
for i in check:
    if lst.count(i) >= 5:
        result.append(i)

if result:
    print(''.join(result))
else:
    print('PREDAJA')