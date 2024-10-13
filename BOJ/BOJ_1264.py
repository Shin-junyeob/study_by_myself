import sys, re
while True:
    string = sys.stdin.readline().rstrip()
    if string == '#':
        exit()
    f = re.findall('[aeiou]', string.lower())
    if f:
        print(len(f))
    else:
        print(0)