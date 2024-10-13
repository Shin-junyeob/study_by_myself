import sys
from itertools import combinations

l, c = map(int, sys.stdin.readline().split())
lst = [x for x in sys.stdin.readline().split()]
lst2 = ['a','e','i','o','u']
set1 = sorted(list(set(lst) - set(lst2)))
set2 = sorted(list(set(lst) & set(lst2)))

def func(lst, l):
    com = list(combinations(lst, l))
    ans = []
    for i in range(len(com)):
        if (len(set(set2) & set(com[i])) >= 1) and (len(set(set1) & set(com[i])) >= 2):
            ans.append(sorted(com[i]))

    real_ans = []
    for i in ans:
        word = ''
        for j in range(l):
            word += i[j]
        real_ans.append(word)
    return sorted(real_ans)

final = func(lst, l)

for i in final:
    print(i)