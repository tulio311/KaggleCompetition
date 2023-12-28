import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt

X = pd.read_csv("train_essays.csv")

#print(X)

textos = X.text
labels = X.generated

n = len(textos)

arr = ['NNS', '.', 'VBP', 'VBN', 'IN', 'PRP', 'VBD', 'JJ', 'DT', 'CD', ',', 'WRB', 'NNP', 'CC', 'NN', 'PRP$', 'VBZ', 'RB', 'VBG', 'TO', 'VB', 'MD', '``', "''", ':', 'NNPS', 'PDT', 'WDT', 'RP', 'JJR', 'JJS', 'RBR', 'EX', 'POS', 'RBS', 'WP', 'UH', 'FW', '$', 'WP$']
numAt = len(arr)

M = np.eye(n,numAt)

for r in range(n):
    print(r)
    v = nltk.tokenize.word_tokenize(textos[r])
    m = len(v)
    v1 = nltk.tag.pos_tag(v)
    k = np.zeros(numAt)
    
    for var in v1:
        for j in range(numAt):
            if var[1] == arr[j]:
                k[j] += 1/m 
    M[r] = k
    

    #print(len(v1) - sum(k))

    #if labels[r] == 1:
    #    plt.plot(np.arange(0,numAt),k,color="red")
    #if labels[r] == 0:
    #    plt.plot(np.arange(0,numAt),k,color="blue")

arrf = np.zeros(numAt)
for i in range(numAt):
    sum = 0
    cont = 0
    for j in range(n):
        if labels[j] == 1:
            sum += M[j][i]
            cont += 1
    arrf[i] = sum/cont



plt.plot(np.arange(0,numAt),arrf, color = "red")
plt.show()


#print(v1)

#arr = ['CC','CD','CS','EX','IN','JJ','JJA','JJC','JJCC','JJS','JJF','JJM','NN','NNA','NNC','NNS','NNP','NNPC','PRP','PRPS','PRP$','RB','RBR','RBS','VB','VBA','VBD','VBG','VBN','VBZ','FW','SYM','PUN']




#print(k)
#print(len(v1))
#print(sum(k))

#print(np.arange(0,numAt))

#plt.plot(np.arange(0,numAt),k)
#plt.show()

#for i in range(n):
#    for j in range(33):
#        M[i][j]
    



