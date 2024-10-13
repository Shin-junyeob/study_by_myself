import sys
n, m = map(int, sys.stdin.readline().split())
ans = [0 for x in range(n)]

for i in range(n):
    p, l = map(int, sys.stdin.readline().split())
    lst = [int(x) for x in sys.stdin.readline().split()]
    if p >= l:
        ans[i] = sorted(lst)[-l]
    else:
        ans[i] = 1

ans = sorted(ans)

while sum(ans) > m:
    ans.pop()

print(len(ans))