lst = []

n = int(input())
for i in range(n):
    lst.append(int(input()))

lst.sort()
for i in range(len(lst)):
    print(lst[i])