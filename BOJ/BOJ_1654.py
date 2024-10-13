import sys
k, n = map(int, sys.stdin.readline().split())
lst = []
for _ in range(k):
    lst.append(int(sys.stdin.readline()))

start = 1
end = max(lst)
result = []
while start <= end:
    pivot = (start+end)//2
    cnt = 0
    for i in lst:
        cnt += i // pivot
    if cnt < n:
        end = pivot - 1
    elif cnt >= n:
        result.append(pivot)
        start = pivot + 1

print(max(result))