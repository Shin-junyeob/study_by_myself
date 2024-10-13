import sys
n, m = map(int, sys.stdin.readline().split())

l = set()
for i in range(n):
    l.add(sys.stdin.readline().rstrip())

s = set()
for i in range(m):
    s.add(sys.stdin.readline().rstrip())

ans = sorted(list(l&s))

print(len(ans))
for i in ans:
    print(i)