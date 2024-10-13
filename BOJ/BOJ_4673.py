def d(num):
    ans = []
    for i in range(1, num):
        lst = [int(x) for x in str(i)]
        ans.append(i + sum(lst))
    return ans

a = d(10000)
b = [int(x) for x in range(1,10001)]

for i in a:
    if i in b:
        b.remove(i)

for i in b:
    print(i)