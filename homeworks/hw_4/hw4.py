# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 10:36:12 2016

@author: Chetan
"""

import numpy as np
from nltk.corpus import brown
from nltk.corpus import stopwords
from collections import Counter
import string
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sps
#-------------------------Utilities----------------------------#
def getTopNWordsFromBrownCorpus(n, context_size):
    all_words = brown.words()
    words = []
    cachedStopWords = stopwords.words("english")
    #operators = set(('and', 'or', 'not', 'the')) #Keeping to remove additional useless words 
    # just add a "not currWord in operators" in the if condition below    
    
    for i in range(len(all_words)):
        currWord = all_words[i].lower()
        if (not (currWord in cachedStopWords) 
            and not (currWord in string.punctuation)):
            for punc in string.punctuation:
                currWord = currWord.replace(punc, "")
            words.append(currWord)

    contextWords, vocab = getTopN(words, n, context_size)     
                     
    return words, vocab, contextWords
    
def getTopN(words, vocabSize, contextSize):
    cnt = Counter(words)
    vocab = []
    contextWords = []
    for k, v in cnt.most_common(vocabSize):
        vocab.append(k)
    for k, v in cnt.most_common(contextSize):
        contextWords.append(k)
    
    return contextWords, vocab
    
def getWordCountMatrix(allWords, vocab, contextWords, totalWordCount):
    wordMatrix = np.zeros(shape = (vocab_size, context_size))
    
    for i in range(totalWordCount):
        word = allWords[i]
        if (word in vocab):
            vocabIndex = vocab.index(word)
            contextIndex = -1
            if (i-1 >= 0 and allWords[i-1] in contextWords):
                contextIndex = contextWords.index(allWords[i-1])
                wordMatrix[vocabIndex][contextIndex] += 1
            if (i-2 >= 0 and allWords[i-2] in contextWords):
                contextIndex = contextWords.index(allWords[i-2])
                wordMatrix[vocabIndex][contextIndex] += 1
            if (i+1 < totalWordCount and allWords[i+1] in contextWords):
                contextIndex = contextWords.index(allWords[i+1])
                wordMatrix[vocabIndex][contextIndex] += 1
            if (i+2 < totalWordCount and allWords[i+2] in contextWords):
                contextIndex = contextWords.index(allWords[i+2])
                wordMatrix[vocabIndex][contextIndex] += 1
    return wordMatrix

def getCWProbMatrix(wordMatrix):
    cwProbMatrix = np.zeros(shape=(len(wordMatrix), len(wordMatrix[0])))
    
    for i in range(len(wordMatrix)):
        rowSum = 0
        for j in range(len(wordMatrix[0])):
            rowSum += (wordMatrix[i][j])
        for j in range(len(wordMatrix[0])):
            cwProbMatrix[i][j] = (wordMatrix[i][j]/rowSum)
    
    return cwProbMatrix
    
def getCiProbList(wordMatrix):
    PrCi = np.zeros(len(wordMatrix[0]))
    
    colSum = []
    totalSum = 0    
    
    for j in range(len(wordMatrix[0])):    
        colSum1 = 0
        for i in range(len(wordMatrix)):
            colSum1 += wordMatrix[i][j]
            totalSum += wordMatrix[i][j]
        colSum.append(colSum1)
    
    for i in range(len(wordMatrix[0])):
        PrCi[i] = (colSum[i]/totalSum)
    
    return PrCi
    
def getPPMIMatrix(PrCW, PrCi):
    PPMIMatrix = np.zeros(shape = (len(PrCW), len(PrCW[0])))
    
    for i in range(len(PrCW)):
        for j in range(len(PrCW[0])):
            ratio = PrCW[i][j]/PrCi[j]
            if (ratio >= 1):
                PPMIMatrix[i][j] = np.log(ratio)
    
    return PPMIMatrix
    
def selectKBestFeatures(PPMIMatrix, k):
    u, s, v = np.linalg.svd(PPMIMatrix, full_matrices = True)
    s = np.diag(s)
    u1 = u[:,:k] #extract first 100 columns
    s = s[:,:k] #extract first 100 columns
    s = s[:k]  #s is diagonal, so restrict to the first 100 rows
    
    reducedSizeMatrix = np.matmul(u1, s) # or simple u1*s?
    return reducedSizeMatrix
    
def runNearestNeighbors(words_to_test, PPMIMatrix, vocab):
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='brute', metric = 'cosine').fit(PPMIMatrix) 
    indices = nbrs.kneighbors(PPMIMatrix, return_distance=False)
    for word in words_to_test:
        index = vocab.index(word)
        nnWord = indices[index][1]
        print (word, "-", vocab[nnWord])
    return
    
def runKMeans(reducedSizeMatrix):
    kmeans = KMeans(n_clusters=100, init='k-means++', max_iter=100000, tol=0.0001, n_jobs=-1).fit(reducedSizeMatrix)
    labels = kmeans.labels_
    category = 100
    filename = "kmeansResults"
    targetList = []

    for i in range(category):
        cat= i+1
        target = open(filename+str(cat)+".txt", 'w+')
        targetList.append(target)
        
    for i in range(len(labels)):
        currLabel = labels[i]
        targetList[currLabel].write(vocab[i])
        targetList[currLabel].write("\n")
    
    for i in range(category):
        targetList[i].close()        
    return "kMeans was successful"

#--------------------------Main--------------------------------#
if __name__ == "__main__":
    
    vocab_size = 5000
    context_size = 1000    

    allWords, vocab, contextWords = getTopNWordsFromBrownCorpus(vocab_size, context_size)

    totalWordCount = len(allWords)
    wordMatrix = getWordCountMatrix(allWords, vocab, contextWords, totalWordCount)
        
    # This matrix gives Pr(c|w)
    PrCW = getCWProbMatrix(wordMatrix)
    # This vector gives Pr(c)
    PrCi = getCiProbList(wordMatrix)
    
    # find the positive pointwise mutual information 
    PPMIMatrix = getPPMIMatrix(PrCW, PrCi)
    reducedSizeMatrix = selectKBestFeatures(PPMIMatrix, 100)
    
    # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    # KMeans using EM algorithm is the input matrix is sparse, and Elkan algorithm otherwise
    if (sps.issparse(reducedSizeMatrix)):
        print ("EM algorithm")
    else:
        print ("Elkan")

    # This list contains 14 + 25 words    
    words_to_test = ['communism', 'autumn', 'cigarette', 'pulmonary', 'mankind',
                        'africa', 'chicago', 'revolution', 'september', 'chemical',
                        'detergent', 'dictionary', 'storm', 'worship', #Given words till here
                        'artistic', 'dogs', 'temple', 'lieutenant', 'michelangelo',
                        'democrats', 'advertising', 'missile', 'rehabilitation', 'electronics',
                        'catholic', 'science', 'literature', 'knowledge', 'civilization',
                        'shoulders', 'santa', 'administrative', 'painting', 'chlorine',
                        'weapon', 'vision', 'submarines', 'institution', 'atlantic']    
    
    # run k-means
    print (runKMeans(reducedSizeMatrix))
        
    # run nearest neighbors
    print ("original")
    runNearestNeighbors(words_to_test, PPMIMatrix, vocab)
    print ("reduced")
    runNearestNeighbors(words_to_test, reducedSizeMatrix, vocab)