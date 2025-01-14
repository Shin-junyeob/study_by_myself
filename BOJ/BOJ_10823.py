<<<<<<< HEAD
string = ''
while True:
    try:
        string += input()
    except: break

=======
string = ''
while True:
    try:
        string += input()
    except: break

>>>>>>> 77e016886 (Initial commit)
print(sum(map(int, string.split(','))))