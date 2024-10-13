import sys
l = int(sys.stdin.readline())
s = sorted([int(x) for x in sys.stdin.readline().split()])
n = int(sys.stdin.readline())
if n in s:
    print(0)
    exit()
small = 0
big = 0
while big == 0:
    for i in s:
        if i < n:
            small = i
        if i > n:
            big = i
            break

small_list = [int(x) for x in range(small+1, n+1)]
big_list = [int(x) for x in range(n, big)]

count = 0
for i in small_list:
    for j in big_list:
        if i != j:
            count += 1
print(count)