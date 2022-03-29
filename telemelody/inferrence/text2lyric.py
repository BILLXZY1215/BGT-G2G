import re
f = open('input_zh.txt','r')
Lines = f.readlines()
res = ""
for Line in Lines:
    L = re.sub('\n', '', Line)
    res = res + L.replace("", " ")[1: -1].replace("，", "[sep]").replace("。","[sep]") + " [sep] "

print(res)