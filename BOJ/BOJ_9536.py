import sys
t = int(sys.stdin.readline())


for _ in range(t):
    lst = []
    string = sys.stdin.readline().rstrip().split()
    while True:
        sound = sys.stdin.readline().rstrip()
        if sound == 'what does the fox say?':
            break
        lst.append(sound.split(' goes ')[1])

    answer = []
    for str in string:
        if str not in lst:
            answer.append(str)
    print(' '.join(answer))