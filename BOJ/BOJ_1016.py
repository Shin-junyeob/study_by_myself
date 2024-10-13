import sys
min, max = map(int, sys.stdin.readline().split())
check = [False] * (max-min+1)
answer = max - min + 1

for i in range(2, int(max**0.5+1)):
    for j in range((((min-1)//(i**2)+1)*i**2), max+1, i**2):
        if not check[j-min]:
            check[j-min] = True
            answer -= 1
print(answer)