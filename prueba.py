import pandas as pd
import numpy as np
import nltk

X = pd.read_csv("train_essays.csv")

#print(X)

textos = X.text
labels = X.generated

n = len(textos)

a = np.zeros(33)

#print(textos[0])

v = nltk.tokenize.word_tokenize(textos[0])
v1 = nltk.tag.pos_tag(v)

print(v1)

L = []

#for w in v1:
#    if w[1] not in L:
#        L.append(w[1])

#for i in range(n):
#    v = nltk.tokenize.word_tokenize(textos[i])
#    v1 = nltk.tag.pos_tag(v)
#    for w in v1:
#        if w[1] not in L:
#            L.append(w[1])
#    print(i)
        

#print(L)        


#sum=0
#for i in range(n):
#    if labels[i] == 1:
#        sum += 1

#print(sum)



