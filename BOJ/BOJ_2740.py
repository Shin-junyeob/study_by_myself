import sys

n, m = map(int, sys.stdin.readline().split())
A = [list(map(int, sys.stdin.readline().rstrip().split())) for _ in range(n)]
m, k = map(int, sys.stdin.readline().split())
B = [list(map(int, sys.stdin.readline().rstrip().split())) for _ in range(m)]

answer = [[0 for _ in range(k)] for _ in range(n)]

for i in range(n):
    for j in range(k):
        for l in range(m):
            answer[i][j] += A[i][l] * B[l][j]

for i in answer:
    for j in i:
        print(j, end = ' ')
    print()