# This code has the central model proposal, it uses the nltk functions of tokenizing and tagging words  

import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt

X = pd.read_csv("train_essays.csv")

textos = X.text
labels = X.generated

n = len(textos)

# These are the tags used by the tag.pos_tag function of nltk
arr = ['NNS', '.', 'VBP', 'VBN', 'IN', 'PRP', 'VBD', 'JJ', 'DT', 'CD', ',', 'WRB', 'NNP', 'CC', 'NN', 'PRP$', 'VBZ', 'RB', 'VBG', 'TO', 'VB', 'MD', '``', "''", ':', 'NNPS', 'PDT', 'WDT', 'RP', 'JJR', 'JJS', 'RBR', 'EX', 'POS', 'RBS', 'WP', 'UH', 'FW', '$', 'WP$']
numAt = len(arr)

M = np.eye(n,numAt)

# We tokenize and tag every text and then calculate the proportion of each tag in the text, then we create a vector with such proportions
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
    
# Matrix M has the proportions of each text in each row

# In the next code we take averages of each word type in AI written texts. We can modify the 1 in labels[j] == 1: 
# to calculate averages in human written texts. We then plot these averages, this may helps us to fand any patterns
# in this proportions of word types so that we can distinguish human written text from AI written text.

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





