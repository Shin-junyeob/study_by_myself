import sys

n, m = map(int, sys.stdin.readline().split())
s = [0 for _ in range(n)]
for i in range(n):
    s[i] = sys.stdin.readline().rstrip()

words = []
for i in range(m):
    word = sys.stdin.readline().rstrip()
    if word in s:
        words.append(word)

print(len(words))