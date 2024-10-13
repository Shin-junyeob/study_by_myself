import sys
n, m = map(int, sys.stdin.readline().split())
dic = dict()
for _ in range(n):
    site, password = sys.stdin.readline().rstrip().split()
    dic[site] = password

for _ in range(m):
    print(dic[sys.stdin.readline().rstrip()])