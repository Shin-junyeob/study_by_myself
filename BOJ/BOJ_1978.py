import sys
n = int(sys.stdin.readline())
lst = list(map(int, sys.stdin.readline().split()))
check = [False, False] + [True]*999
primes = []
answer = 0
for i in range(2,1001):
    if check[i]:
        primes.append(i)
        for j in range(2*i, 1001, i):
            check[j] = False

for i in lst:
    if i in primes:
        answer += 1
print(answer)