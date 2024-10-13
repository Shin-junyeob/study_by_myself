lst = []
for i in range(int(input())):
    a, b = map(int, input().split())
    lst.append(a+b)

for i in range(len(lst)):
    print(lst[i])