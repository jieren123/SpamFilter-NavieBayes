## Project Overview
An implementation of a Naive Bayesian Classifier in Python. Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of independence between every pair of features.This methods to classify documents, based on the words that appear within them. A common application for this type of software is in email spam filters.

## 1. Navie Bayes Spam Filtering
S =  the probability that the email is spam
H =  the probability that the email is ham
```
Background: Bayes Therom P(S|W) = (P(W|S)*P(S))/(P(W|S)*P(S)+P(W|H)*P(H))

```
S = the email is spam, H = the email is ham, W = the word 
- Pr(S|W) is the probability that a message is a spam, knowing that the word is in it;
- Pr(S) is the overall probability that any given message is spam;
- Pr(W|S) is the probability that the word "replica" appears in spam messages;
- Pr(H) is the overall probability that any given message is not spam (is "ham");
- Pr(W|H) is the probability that the word "replica" appears in ham messages.

Combining individual probabilities

p = (p1 * p2 ... pn) / ( (p1 * p2 ... * pn) + ( (1 - p1) * (1 - p2) ... * (1 - pn) ) )
where:
- p is the probability that the suspect message is spam
- p1 is the probability P(S|w_1) that it is a spam knowing it contains a first spam word
- p2 is the probability P(S|W_2) that it is a spam knowing it contains a second spam word 
......
- pn is the probability p(S|W_N) that it is a spam knowing it contains an Nth word
 
The result p is typically compared to a given threshold to decide whether the message is spam or not. If p is lower than the threshold, the message is considered as likely ham, otherwise it is considered as likely spam

**But the disadvantage of this suitation**
*Rare word* : When the word doesn't exist, both the numerator and the denominator are equal to zero. Or more generally the words that were encountered only a few times during the learning phase cause a problem, because it would be an error to trust blindly the informatio n they provide.
-Solution: 
- 1. Pr(S) can again be taken equal to 0.5, to avoid being too suspicious about incoming email.
```
wordsInSpamNum = np.ones(numWords)
wordsInHealthNum = np.ones(numWords)
spamWordsNum = 2.0
healthWordsNum = 2.0
```
- 2. Take log of P(Wi|S) and P(Wi|H)
```
pWordsSpamicity = np.log(wordsInSpamNum / spamWordsNum)
pWordsHealthy = np.log(wordsInHealthNum / healthWordsNum)
```

## 2 Running Adaboost on Naive Bayes to Prevent Bayesian Poisoning
AdaBoost, (Adaptive Boosting) is a boosting approach in machine learning based on the idea of creating a highly accurate prediction rule by combining many relatively weak and inaccurate rules. I implemented AdaBoost and NaiveBayes as a classifier interface in my framework in order to increase the spamicity rate of word.
```
	set iteratenum = 1000 
    for i in range(iterateNum):
        set errorCount = 0.0
        for j in range(testCount):
            if Type != testWordsType[j]:
                errorCount += 1
                alpha = ps - ph
                if alpha > 0: # orginal: ham，predict： spam
                    DS[testWordsCount != 0] = np.abs(
                            (DS[testWordsCount != 0] - np.exp(alpha)) / DS[testWordsCount != 0])
                else:  # original: spam，predict: ham
                    DS[testWordsCount != 0] = (DS[testWordsCount != 0] + np.exp(alpha)) / DS[testWordsCount != 0]
                return DS

Get DS to adjust ps

    ps = sum(testWordsMarkedArray * pWordsSpamicity * DS) + np.log(pSpam)

```

You can import AdaBoostClassifier from sklearn.ensemble for this part. I write it just in order to full presently my idea.

## Reference 
- [Naive Bayes spam filtering](https://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering)
- [AdaBoost](https://en.wikipedia.org/wiki/AdaBoost)
- [Boosting (machine learning)](https://en.wikipedia.org/wiki/Boosting_(machine_learning))
- [Boosting and naive bayesian learning](http://pages.cs.wisc.edu/~dyer/cs540/handouts/elkan97boosting.pdf)
- [Running Adaboost on Naive Bayes](http://web.cecs.pdx.edu/~mm/MachineLearningWinter2010/BoostingNaiveBayes.pdf)
