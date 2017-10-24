## Project Overview
An implementation of a Naive Bayesian Classifier in Python. Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of independence between every pair of features.This methods to classify documents, based on the words that appear within them. A common application for this type of software is in email spam filters.

## 1. Navie Bayes Spam Filtering
S =  the probability that the email is spam
H =  the probability that the email is ham
```
Background: Bayes Therom P(S|W) = (P(W|S)*P(S))/(P(W|S)*P(S)+P(W|H)*P(H))

```

## 2. Adaptive Boosting on Navie Bayes
AdaBoost, (Adaptive Boosting) is a boosting approach in machine learning based on the idea of creating a highly accurate prediction rule by combining many relatively weak and inaccurate rules. I implemented AdaBoost and NaiveBayes as a classifier interface in my framework, however, AdaBoost doesn’t have its own classification algorithm, rather it runs the NaiveBayes algorithm several times to create multiple weak classifiers which are then combined based on their error rates in order to improve the overall performance of the NaiveBayes classifier.


## Reference 
- [Naive Bayes spam filtering](https://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering)
- [AdaBoost](https://en.wikipedia.org/wiki/AdaBoost)
- [Boosting (machine learning)](https://en.wikipedia.org/wiki/Boosting_(machine_learning))
