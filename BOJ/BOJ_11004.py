N, K = map(int, input().split())

lst = input().split()

for i in range(N):
    lst[i] = int(lst[i])

lst.sort()

print(lst[K-1])