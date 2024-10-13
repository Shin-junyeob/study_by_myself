import sys
n = int(sys.stdin.readline())
lst = []
check = [[] for _ in range(n)]
for _ in range(n):
    word = sys.stdin.readline().rstrip()
    lst.append(word)

cnt = 0
for i in range(n):
    for j in lst[i]:
        if j not in check[i]:
            check[i].append(j)
        elif j in check[i] and check[i][-1] != j:
            cnt += 1
            break
        else:
            pass

print(n - cnt)