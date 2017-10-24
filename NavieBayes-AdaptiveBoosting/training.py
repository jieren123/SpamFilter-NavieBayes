#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

import AdaSpamFilter as boostNaiveBayes
import random
import numpy as np


def trainingAdaboostGetDS(iterateNum=40):
    """
    :param iterateNum:
    :return:
    """
    filename = 'training.txt'
    smsWords, classLables = boostNaiveBayes.loadSMSData(filename)

 
    testWords = []
    testWordsType = []

    testCount = 1000
    for i in range(testCount):
        randomIndex = int(random.uniform(0, len(smsWords)))
        testWordsType.append(classLables[randomIndex])
        testWords.append(smsWords[randomIndex])
        del (smsWords[randomIndex])
        del (classLables[randomIndex])


    vocabularyList = boostNaiveBayes.createVocabularyList(smsWords)
    print "Create Vocabulary List"
    trainMarkedWords = boostNaiveBayes.setOfWordsListToVecTor(vocabularyList, smsWords)
    print "Complete Mark word-vector"
    trainMarkedWords = np.array(trainMarkedWords)
    print "Complete word matrix"
    pWordsSpamicity, pWordsHealthy, pSpam = \
        boostNaiveBayes.trainingNaiveBayes(trainMarkedWords, classLables)

    DS = np.ones(len(vocabularyList))

    ds_errorRate = {}
    minErrorRate = np.inf
    for i in range(iterateNum):
        errorCount = 0.0
        for j in range(testCount):
            testWordsCount = boostNaiveBayes.setOfWordsToVecTor(vocabularyList, testWords[j])
            ps, ph, smsType = boostNaiveBayes.classify(pWordsSpamicity, pWordsHealthy,
                                                       DS, pSpam, testWordsCount)

            if smsType != testWordsType[j]:
                errorCount += 1
                # alpha = (ph - ps) / ps
                alpha = ps - ph
                if alpha > 0: # orginal: ham，predict： spam
                    DS[testWordsCount != 0] = np.abs(
                            (DS[testWordsCount != 0] - np.exp(alpha)) / DS[testWordsCount != 0])
                else:  # original: spam，predict: ham
                    DS[testWordsCount != 0] = (DS[testWordsCount != 0] + np.exp(alpha)) / DS[testWordsCount != 0]
        print 'DS:', DS
        errorRate = errorCount / testCount
        if errorRate < minErrorRate:
            minErrorRate = errorRate
            ds_errorRate['minErrorRate'] = minErrorRate
            ds_errorRate['DS'] = DS
        print 'iterate %d ，error number %d ，error rate %f' % (i, errorCount, errorRate)
        if errorRate == 0.0:
            break
    ds_errorRate['vocabularyList'] = vocabularyList
    ds_errorRate['pWordsSpamicity'] = pWordsSpamicity
    ds_errorRate['pWordsHealthy'] = pWordsHealthy
    ds_errorRate['pSpam'] = pSpam
    return ds_errorRate


if __name__ == '__main__':
    dsErrorRate = trainingAdaboostGetDS()
    # save model information 
    np.savetxt('pWordsSpamicity.txt', dsErrorRate['pWordsSpamicity'], delimiter='\t')
    np.savetxt('pWordsHealthy.txt', dsErrorRate['pWordsHealthy'], delimiter='\t')
    np.savetxt('pSpam.txt', np.array([dsErrorRate['pSpam']]), delimiter='\t')
    np.savetxt('trainDS.txt', dsErrorRate['DS'], delimiter='\t')
    np.savetxt('trainMinErrorRate.txt', np.array([dsErrorRate['minErrorRate']]), delimiter='\t')
    vocabulary = dsErrorRate['vocabularyList']
    fw = open('vocabularyList.txt', 'w')
    for i in range(len(vocabulary)):
        fw.write(vocabulary[i] + '\t')
    fw.flush()
    fw.close()
