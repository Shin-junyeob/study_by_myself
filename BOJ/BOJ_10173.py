<<<<<<< HEAD
import sys, re
while True:
    string = sys.stdin.readline().rstrip()
    if string == 'EOI':
        exit()
    m = re.findall('nemo', string.lower())
    if m:
        print('Found')
    else:
=======
import sys, re
while True:
    string = sys.stdin.readline().rstrip()
    if string == 'EOI':
        exit()
    m = re.findall('nemo', string.lower())
    if m:
        print('Found')
    else:
>>>>>>> 77e016886 (Initial commit)
        print('Missing')