import sys
t = int(sys.stdin.readline())
for _ in range(t):
    n = int(sys.stdin.readline())
    lst = []
    for _ in range(n):
        lst.append(sys.stdin.readline().rstrip())
    lst.sort()

    answer = 'YES'
    for i in range(len(lst)-1):
        if lst[i] == lst[i+1][:len(lst[i])]:
            answer = 'NO'

    print(answer)