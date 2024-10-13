n = int(input())
lst = []
for i in range(n):
    a, b = map(int, input().split())
    lst.append([b, a])

lst.sort()

for i in range(n):
    print(lst[i][1], lst[i][0])