import sys
string = sys.stdin.readline().rstrip()
dic = dict(zip(['apa','epe','ipi','opo','upu'], ['a','e','i','o','u']))
for i in dic:
    string = string.replace(i, dic[i])

print(string)