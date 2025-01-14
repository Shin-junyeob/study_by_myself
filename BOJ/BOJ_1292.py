<<<<<<< HEAD
a, b = map(int, input().split())
lst = []
n = 0

for i in range(1, b+1):
    n += i
    while len(lst) != n:
        lst.append(i)

=======
a, b = map(int, input().split())
lst = []
n = 0

for i in range(1, b+1):
    n += i
    while len(lst) != n:
        lst.append(i)

>>>>>>> 77e016886 (Initial commit)
print(sum(lst[a-1:b]))