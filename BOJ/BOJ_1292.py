a, b = map(int, input().split())
lst = []
n = 0

for i in range(1, b+1):
    n += i
    while len(lst) != n:
        lst.append(i)

print(sum(lst[a-1:b]))