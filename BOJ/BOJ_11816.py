import sys, re
x = sys.stdin.readline().rstrip()
f = re.findall('\w', x)
alpha = ['a', 'b', 'c', 'd', 'e', 'f']; num = [10, 11, 12, 13, 14, 15]
dic = dict(zip(alpha, num))
if 'x' in f:
    h = re.split('x', x)
    ans = 0
    for i in range(len(h[1:][0])):
        try:
            ans += int(h[1:][0][i]) * (16**((len(h[1:][0])-i)-1))
        except:
            ans += int(dic[h[1:][0][i]]) * (16**((len(h[1:][0])-i)-1))
    print(ans)
else:
    if f[0] == '0':
        ans = 0
        for i in range(len(x[1:])):
            ans += int(x[1:][i]) * (8**((len(x[1:])-i)-1))
        print(ans)
    else:
        print(int(x))