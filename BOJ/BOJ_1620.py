import sys
n, m = map(int, sys.stdin.readline().split())
dic = {}
for i in range(1, n+1):
    name = sys.stdin.readline().rstrip()
    dic[i] = name
    dic[name] = i

for i in range(m):
    find = sys.stdin.readline().rstrip()
    try:
        print(dic[find])
    except:
        print(dic[int(find)])