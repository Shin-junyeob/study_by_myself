n, x = map(int, input().split())
num = input().split()

for i in range(n):
    if int(num[i]) < x:
        print(num[i], end=" ")