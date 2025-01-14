<<<<<<< HEAD
n = int(input())
lst = []
for i in range(n):
    lst.append(input().split())

for i in range(n):
    lst[i][0] = int(lst[i][0])
    lst[i][1] = int(lst[i][1])

lst.sort()
for i in range(len(lst)):
=======
n = int(input())
lst = []
for i in range(n):
    lst.append(input().split())

for i in range(n):
    lst[i][0] = int(lst[i][0])
    lst[i][1] = int(lst[i][1])

lst.sort()
for i in range(len(lst)):
>>>>>>> 77e016886 (Initial commit)
    print(lst[i][0], lst[i][1])