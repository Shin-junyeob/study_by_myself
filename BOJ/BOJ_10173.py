import sys, re
while True:
    string = sys.stdin.readline().rstrip()
    if string == 'EOI':
        exit()
    m = re.findall('nemo', string.lower())
    if m:
        print('Found')
    else:
        print('Missing')