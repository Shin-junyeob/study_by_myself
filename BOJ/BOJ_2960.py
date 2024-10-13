import sys
n, k = map(int, sys.stdin.readline().split())
lst = [int(x) for x in range(2, n+1)]

count = 0
def func(lst):
    global count
    p = lst[0]
    for x,i in enumerate(lst):
        if i % p == 0:
            lst.remove(i)
            count += 1
            if count == k:
                print(i)
                exit()
    if count < k:
        func(lst)

func(lst)