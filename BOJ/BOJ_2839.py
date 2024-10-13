n = int(input())
big = n // 5
small = (n - 5 * big) // 3
other = n - 5 * big - 3 * small
ans = 0

for i in range(n):
    if other != 0:
        big -= 1
        small += (other+5)//3
        other = n- 5 * big - 3 * small
        if big < 0:
            big = 0; small = -1
    else:
        ans = big + small

print(ans)