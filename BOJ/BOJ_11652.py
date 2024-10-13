import sys
n = int(sys.stdin.readline())
dic = dict()
for i in range(n):
    num = int(sys.stdin.readline().rstrip())
    if num not in dic:
        dic[num] = 1
    else:
        dic[num] += 1

lst = []
for i in dic:
    lst.append((i, dic[i]))

lst.sort(key = lambda x: x[1], reverse = True)
mmax = 0
check = []
for i in lst:
    if i[1] >= mmax:
        mmax = i[1]
        check.append(i[0])
    else:
        break

print(min(check))