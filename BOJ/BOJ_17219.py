<<<<<<< HEAD
import sys
n, m = map(int, sys.stdin.readline().split())
dic = dict()
for _ in range(n):
    site, password = sys.stdin.readline().rstrip().split()
    dic[site] = password

for _ in range(m):
=======
import sys
n, m = map(int, sys.stdin.readline().split())
dic = dict()
for _ in range(n):
    site, password = sys.stdin.readline().rstrip().split()
    dic[site] = password

for _ in range(m):
>>>>>>> 77e016886 (Initial commit)
    print(dic[sys.stdin.readline().rstrip()])