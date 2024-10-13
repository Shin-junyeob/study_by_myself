import sys
m = int(sys.stdin.readline())
n = int(sys.stdin.readline())
check = [False, False] + [True]*(n-1)
primes = []
for i in range(2, n+1):
    if check[i]:
        primes.append(i)
        for j in range(i*2, n+1, i):
            check[j] = False

answer = []
for i in primes:
    if m <= i <= n:
        answer.append(i)

if answer:
    print(sum(answer))
    print(min(answer))
else:
    print(-1)