import numpy as np
import SimpleNavieBayes.NavieBayes as naiveBayes

filename = 'training.txt'
smsWords, classLables = naiveBayes.loadSMSData(filename)

vocabularyList = naiveBayes.createVocabularyList(smsWords)
print "Create Vocabulary List"

trainMarkedWords = naiveBayes.setOfWordsListToVecTor(vocabularyList, smsWords)
print "Complete Mark word-vector"

trainMarkedWords = np.array(trainMarkedWords)
print "Complete word matrix"

pWordsSpamicity, pWordsHealthy, pSpam = naiveBayes.trainingNaiveBayes(trainMarkedWords, classLables)
print 'pSpam:', pSpam

fpSpam = open('pSpam.txt', 'w')
spam = pSpam.__str__()
fpSpam.write(spam)
fpSpam.close()

fw = open('vocabularyList.txt', 'w')
for i in range(len(vocabularyList)):
    fw.write(vocabularyList[i] + '\t')
fw.flush()
fw.close()
np.savetxt('pWordsSpamicity.txt', pWordsSpamicity, delimiter='\t')
np.savetxt('pWordsHealthy.txt', pWordsHealthy, delimiter='\t')
