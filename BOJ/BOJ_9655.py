n = int(input())

count = 0
SK = 0
CY = 0
while True:
    if n != 0:
        if count % 2 == 0:
            if (n % 3 == 0) & (n // 2 != 0):
                n -= 3
                SK += 1
                count += 1
            else:
                n -= 1
                SK += 1
                count += 1
        else:
            if (n % 3 == 0) & (n // 2 == 0):
                n -= 3
                CY += 1
                count += 1
            else:
                n -= 1
                CY += 1
                count += 1
    else:
        break

if SK > CY:
    print('SK')
else:
    print('CY')