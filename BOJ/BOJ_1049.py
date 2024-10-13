import sys
n, m = map(int, sys.stdin.readline().split())
lst = []
for i in range(m):
    p, o = map(int, sys.stdin.readline().rstrip().split())
    lst.append([p, o])

ans1 = (n//6 + 1) * sorted(lst)[0][0]
ans2 = (n//6) * sorted(lst)[0][0] + (n%6) * sorted(lst, key = lambda x:x[1])[0][1]
ans3 = n * sorted(lst, key = lambda x:x[1])[0][1]

print(min(ans1, ans2, ans3))