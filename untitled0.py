S="We test coders. Give us a try?"
S="Forget  CVs..Save time . x x"
Text=""
for i in range(len(S)):
    if(S[i]=='.' or S[i]=='?' or S[i]=='!'):
        Text=Text+'_'
    else:
        Text=Text+S[i]

Sentences=Text.split('_')
maxWords=0
for i in range(len(Sentences)):
    Words=Sentences[i].split(' ')
    words=0
    print(Words)
    for j in range(len(Words)):
        if(Words[j]!=''):
            words+=1
    print(words)
    if(words>maxWords):
        maxWords=words
    
print(maxWords)
S=S.lower()