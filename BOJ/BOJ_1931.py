import sys
n = int(sys.stdin.readline())
lst = []
for i in range(n):
    s, e = map(int, sys.stdin.readline().rstrip().split())
    lst.append([s, e])

lst.sort(key = lambda x:(x[1], x[0]))
print(lst)
cnt = 0
new = 0
for start, end in lst:
    if start >= new:
        cnt += 1
        new = end

print(cnt)