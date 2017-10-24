#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_
import numpy as np
import sys
import os
import re

#  preprocess; get rid of white space; convert upper case to lower case
def textParser(text):
    """
    :param text:
    :return:
    """
    # regext defining what to consider as a word 
    word_regex = re.compile("[a-zA-Z']+(?:-[a-zA-Z']+)?")
    words = word_regex.split(text)
    words = [word.lower() for word in words if len(word) > 0]
    return words

# classify tags，1: trash SMS，0: normal SMS
def loadSMSData(fileName):
    """
    :param fileName:
    :return:
    """
    f = open(fileName)
    classCategory = []  
    smsWords = []
    for line in f.readlines():
        linedatas = line.strip().split('\t')
        if linedatas[0] == 'ham':
            classCategory.append(0)
        elif linedatas[0] == 'spam':
            classCategory.append(1)
        # split text 
        words = textParser(linedatas[1])
        smsWords.append(words)
    return smsWords, classCategory

# build each word phrase
def createVocabularyList(smsWords):
    """
	:param smsWords:
    :return:
    """
    vocabularySet = set([])
    for words in smsWords:
        vocabularySet = vocabularySet | set(words)
    vocabularyList = list(vocabularySet)
    return vocabularyList


def getVocabularyList(fileName):
    """
    :param fileName:
    :return:
    """
    fr = open(fileName)
    vocabularyList = fr.readline().strip().split('\t')
    fr.close()
    return vocabularyList

# find word occurences in messages 
def setOfWordsToVecTor(vocabularyList, smsWords):
    """
    :param vocabularyList:
    :param smsWords:
    :return:
    """
    vocabMarked = [0] * len(vocabularyList)
    for smsWord in smsWords:
        if smsWord in vocabularyList:
            vocabMarked[vocabularyList.index(smsWord)] += 1 #count up occurences of each word in messages

    return vocabMarked

# word frequencies and occurences with words as key
def setOfWordsListToVecTor(vocabularyList, smsWordsList):
    """
    :param vocabularyList:
    :param smsWordsList:
    :return:
    """
    vocabMarkedList = []
    for i in range(len(smsWordsList)):
        vocabMarked = setOfWordsToVecTor(vocabularyList, smsWordsList[i])
        vocabMarkedList.append(vocabMarked)
    return vocabMarkedList


def trainingNaiveBayes(trainMarkedWords, trainCategory):
    """
    spamicity：P（Wi|S）
    :param trainMarkedWords: 
    :param trainCategory:
    :return:
    """
    numTrainDoc = len(trainMarkedWords)
    numWords = len(trainMarkedWords[0])
    # P(S) = Probability in spam emails 
    pSpam = sum(trainCategory) / float(numTrainDoc)

    
    wordsInSpamNum = np.ones(numWords)
    wordsInHealthNum = np.ones(numWords)
    spamWordsNum = 2.0
    healthWordsNum = 2.0
    for i in range(0, numTrainDoc):
        if trainCategory[i] == 1:  # if is spam
            wordsInSpamNum += trainMarkedWords[i]
            spamWordsNum += sum(trainMarkedWords[i]) 
        else:
            wordsInHealthNum += trainMarkedWords[i]
            healthWordsNum += sum(trainMarkedWords[i])
	#gives the conditional probability p(W_i | S_x)
    pWordsSpamicity = np.log(wordsInSpamNum / spamWordsNum)
    pWordsHealthy = np.log(wordsInHealthNum / healthWordsNum)

    return pWordsSpamicity, pWordsHealthy, pSpam


def getTrainedModelInfo():
    """
    :return:
    """
    vocabularyList = getVocabularyList('vocabularyList.txt')
    pWordsHealthy = np.loadtxt('pWordsHealthy.txt', delimiter='\t')
    pWordsSpamicity = np.loadtxt('pWordsSpamicity.txt', delimiter='\t')
    fr = open('pSpam.txt')
    pSpam = float(fr.readline().strip())
    fr.close()

    return vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam

#classifies a new email as spam or not spam
def classify(vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam, testWords):
    """
    :param vocabularyList:
    :param pWordsSpamicity:
    :param pWordsHealthy:
    :param pSpam:
    :param testWords:
    :return:
    """
    testWordsCount = setOfWordsToVecTor(vocabularyList, testWords)
    testWordsMarkedArray = np.array(testWordsCount)
    # calculate P(Ci|W) = (W|Ci)P(Ci)
    p1 = sum(testWordsMarkedArray * pWordsSpamicity) + np.log(pSpam)
    p0 = sum(testWordsMarkedArray * pWordsHealthy) + np.log(1 - pSpam)
    if p1 > p0:
        return 1
    else:
        return 0
