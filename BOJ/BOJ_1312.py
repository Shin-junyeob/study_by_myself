# 정답
A, B, N = map(int, input().split())

for i in range(N):
    A = (A%B)*10

print(A//B)

# 정답인것 같으나 오답
# 부동소수점에 대해서 공부할 필요 있음
A, B, N = map(int, input().split())

ans = f'{A/B:.{N+1}F}'
print(ans[-2])