<<<<<<< HEAD
N, K = map(int, input().split())

lst = input().split()

for i in range(N):
    lst[i] = int(lst[i])

lst.sort()

=======
N, K = map(int, input().split())

lst = input().split()

for i in range(N):
    lst[i] = int(lst[i])

lst.sort()

>>>>>>> 77e016886 (Initial commit)
print(lst[K-1])