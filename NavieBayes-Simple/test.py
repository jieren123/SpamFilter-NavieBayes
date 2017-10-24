#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_


import numpy as np
import random
import SpamFilter as naiveBayes

import random
import numpy as np


def simpleTest():

    vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam = \
        naiveBayes.getTrainedModelInfo()


    filename = '../emails/test/test.txt'
    smsWords, classLables = naiveBayes.loadSMSData(filename)

    smsType = naiveBayes.classify(vocabularyList, pWordsSpamicity,
                                  pWordsHealthy, pSpam, smsWords[0])
    print smsType


def testClassifyErrorRate():
    """
    :return:
    """
    filename = '../emails/training/SMSCollection.txt'
    smsWords, classLables = naiveBayes.loadSMSData(filename)

    testWords = []
    testWordsType = []

    testCount = 1000
    for i in range(testCount):
        randomIndex = int(random.uniform(0, len(smsWords)))
        testWordsType.append(classLables[randomIndex])
        testWords.append(smsWords[randomIndex])
        del (smsWords[randomIndex])
        del (classLables[randomIndex])

    vocabularyList = naiveBayes.createVocabularyList(smsWords)
    print "Create Vocabulary List"
    trainMarkedWords = naiveBayes.setOfWordsListToVecTor(vocabularyList, smsWords)
    print "Complete Mark word-vector"
    trainMarkedWords = np.array(trainMarkedWords)
    print "Complete word matrix"
    pWordsSpamicity, pWordsHealthy, pSpam = naiveBayes.trainingNaiveBayes(trainMarkedWords, classLables)

    errorCount = 0.0
    for i in range(testCount):
        smsType = naiveBayes.classify(vocabularyList, pWordsSpamicity,
                                      pWordsHealthy, pSpam, testWords[i])
        print 'PredictType：', smsType, 'RealType：', testWordsType[i]
        if smsType != testWordsType[i]:
            errorCount += 1

    print 'error-number：', errorCount, 'error-rate：', errorCount / testCount


if __name__ == '__main__':
    simpleTest()
