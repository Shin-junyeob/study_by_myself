import sys
n = int(sys.stdin.readline())

for i in range(n):
    score = 0
    word = sys.stdin.readline().rstrip()
    count = 1
    for i in range(len(word)):
        if word[i] == 'O':
            score += count
            count += 1
        if word[i] == 'X':
            count = 1
    print(score)