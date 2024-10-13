import sys
n , k = map(int, sys.stdin.readline().split())

lst = [0 for x in range(n)]
for i in range(n):
    lst[i] = int(sys.stdin.readline().rstrip())

count = 0
for i in lst[::-1]:
    if i > k:
        pass
    if i <= k:
        count += k // i
        k %= i

print(count)