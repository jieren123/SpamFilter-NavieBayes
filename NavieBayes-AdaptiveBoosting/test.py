#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_


import numpy as np

import AdaSpamFilter as boostNaiveBayes


def getTrainAdaboostInfo():
    """
    :return:
    """
    trainDS = np.loadtxt('trainDS.txt', delimiter='\t')
    trainMinErrorRate = np.loadtxt('trainMinErrorRate.txt', delimiter='\t')
    vocabularyList = boostNaiveBayes.getVocabularyList('vocabularyList.txt')
    pWordsSpamicity = np.loadtxt('pWordsSpamicity.txt', delimiter='\t')
    pWordsHealthy = np.loadtxt('pWordsHealthy.txt', delimiter='\t')
    pSpam = np.loadtxt('pSpam.txt', delimiter='\t')
    return vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam, trainMinErrorRate, trainDS


def simpleTest():
    vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam, trainMinErrorRate, trainDS = \
        getTrainAdaboostInfo()

    filename = 'test.txt'
    smsWords, classLables = boostNaiveBayes.loadSMSData(filename)
    testWordsMarkedArray = \
        boostNaiveBayes.setOfWordsToVecTor(vocabularyList, smsWords[0])
    ps, ph, smsType = boostNaiveBayes.classify(
            pWordsSpamicity, pWordsHealthy, trainDS, pSpam, testWordsMarkedArray)
    print smsType


if __name__ == '__main__':
    simpleTest()
