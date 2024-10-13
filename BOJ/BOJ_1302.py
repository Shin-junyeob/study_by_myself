import sys
n = int(sys.stdin.readline())
dic = dict()
for _ in range(n):
    book = sys.stdin.readline().rstrip()
    if book not in dic:
        dic[book] = 1
    else:
        dic[book] += 1

lst = []
for i in list(dic.keys()):
    lst.append((i, dic[i]))

lst.sort(key = lambda x: x[1], reverse = True)

check = []
answer = 0
for i in lst:
    if i[1] >= answer:
        answer = i[1]
        check.append(i[0])
    else:
        break
check.sort()
print(check[0])