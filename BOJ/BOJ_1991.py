import sys
from collections import defaultdict

n = int(sys.stdin.readline())
dic = defaultdict(list)
for _ in range(n):
    node, left, right = sys.stdin.readline().split()
    dic[node] = [left, right]

answer1 = []
def func1(node):
    if node == '.':
        return

    answer1.append(node)
    func1(dic[node][0])
    func1(dic[node][1])
    return ''.join(answer1)

answer2 = []
def func2(node):
    if node == '.':
        return

    func2(dic[node][0])
    answer2.append(node)
    func2(dic[node][1])
    return ''.join(answer2)

answer3 = []
def func3(node):
    if node == '.':
        return

    func3(dic[node][0])
    func3(dic[node][1])
    answer3.append(node)
    return ''.join(answer3)

print(func1('A'))
print(func2('A'))
print(func3('A'))