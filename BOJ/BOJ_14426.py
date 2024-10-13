import sys

n, m = map(int, sys.stdin.readline().split())
S = []; check = []; answer = 0; idx = 0
for _ in range(n):
    S.append(sys.stdin.readline().rstrip())
S.sort()
for _ in range(m):
    check.append(sys.stdin.readline().rstrip())
check.sort()

for i in check:
    for j in range(idx, n):
        if i == S[j][:len(i)]:
            answer += 1
            idx = j
            break

print(answer)