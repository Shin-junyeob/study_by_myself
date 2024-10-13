string = ''
while True:
    try:
        string += input()
    except: break

print(sum(map(int, string.split(','))))