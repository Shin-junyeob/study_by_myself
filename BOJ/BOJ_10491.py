<<<<<<< HEAD
import re
while True:
    try:
        a=input()
        p=re.match('.*problem', a.lower())
        if p: print('yes')
        else: print('no')
=======
import re
while True:
    try:
        a=input()
        p=re.match('.*problem', a.lower())
        if p: print('yes')
        else: print('no')
>>>>>>> 77e016886 (Initial commit)
    except EOFError: break