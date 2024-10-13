n = input().zfill(2)
m = n[:1] + n[-1:]
cycle = 0
while True:
    m = m[-1:] + str(int(m[:1]) + int(m[-1:]))[-1:]
    cycle += 1
    if n == m:
        break
print(cycle)