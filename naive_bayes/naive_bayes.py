#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

class naive_bayes_spam_filter:


    def __init__(self,reject_rate=15):

        self.reject = reject_rate  # initial reject rate  p(spam|x)/p(ham|x)

    def get_params(self, deep=False):
        return {'reject_rate': self.reject}

    def set_params(self, dict):

        if 'reject_rate' in dict:
            self.reject = dict['reject_rate']

    def fit(self, training_data, labels):
        """
        return matrix of spam/ham for the document
        :param training_data: bags of words   [num_doc, num_words]  training_data[i,j] = frequency of jth word in ith document
        :param labels:  labels [num_doc]  labels[i] = 1/0 spam /ham
        :return self
        """
        # Count ham/spam occurrence of each word
        # optimize it! very slow
        self.n_doc = training_data.shape[0]  # num of documents
        self.p_spam = np.float(np.sum(labels) / self.n_doc)
        self.p_ham = 1 - self.p_spam
        self.n_words = training_data.shape[1]

        ## I could use  n_w_spam = 0 , n_w_ham = 0, and p[:,:] = 0 as
        ## initial value
        ## but later might encouter "divide by zero encountered in log" problem
        ## to give them all offset of n_words avoid it
        self.n_w_spam = self.n_words
        self.n_w_ham =  self.n_words
        # p[0]= p(wi|spam) , p[1]= p(wi|ham)
        self.p = np.ones((2, self.n_words))

        for i in range(self.n_doc):
            if labels[i] == 1:
                self.p[0] += training_data[i]
                self.n_w_spam += np.sum(training_data[i])
            else:
                self.p[1] += training_data[i]
                self.n_w_ham += np.sum(training_data[i])
        # normalize counts to  yield word probabilities
        self.p[0] = self.p[0] / self.n_w_spam
        self.p[1] = self.p[1] / self.n_w_ham
        return self, self.p

    def predict(self, test_data):
        """

        :param test_data: bags of words   [num_doc, num_words]  test_data[i,j] = frequency of jth word in ith document
        :return: prediction matrix
        """

        # initial reject threshold offset  = p(w|spam)/p(w|ham). reject is what can control the precision
        b = np.log(self.reject * self.p_ham / self.p_spam )
        predicts = np.full((test_data.shape[0]), -b)
        for i in range(test_data.shape[0]):
            predicts[i] += test_data[i] * ( np.log(self.p[0]) - np.log(self.p[1]))
            if predicts[i] > 0:
                predicts[i] = 1
            else:
                predicts[i] = 0
        return predicts


if __name__ == "__main__":
    # load the data
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report

    df = pd.read_table('smsspamcollection/SMSSpamCollection',
                       sep='\t',
                       header=None,
                       names=['label', 'sms_message'])
    df['label'] = df.label.map({'ham': 0, 'spam': 1})

    # split to training and testing set
    X_train, X_test, y_train, y_test = train_test_split(df['sms_message'],
                                                        df['label'],
                                                        test_size = 0.25,
                                                        random_state=1)
    print("Train data size: ", len(X_train))
    print("Test data size: ", len(X_test))
    # pre-process the documents to the bags of words
    count_vector = CountVectorizer(stop_words='english')
    #Learn the vocabulary dictionary and return term-document matrix.
    #This is equivalent to fit followed by transform, but more efficiently implemented.
    x_training_data = count_vector.fit_transform(X_train)

    # Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
    testing_data = count_vector.transform(X_test)
    # print(testing_data)
    naive_bayes = MultinomialNB()
    naive_bayes = naive_bayes.fit(x_training_data, y_train)
    predicts = naive_bayes.predict(testing_data)

    print("sklearn prediction: ", predicts)
    print('sklearn Accuracy score: ', format(accuracy_score(y_test, predicts)))
    print('sklearn Precision score: ', precision_score(y_test, predicts))
    print('sklearn Recall score: ', format(recall_score(y_test, predicts)))
    print('sklearn F1 score: ', format(f1_score(y_test, predicts)))

    # compare against my own
    # has to be all numpy matrix
    spam_filter = naive_bayes_spam_filter()
    spam_filter.fit(x_training_data,y_train.as_matrix())

    predicts = spam_filter.predict(testing_data)
    print("my prediction: ", predicts)
    print('my Accuracy score: ', format(accuracy_score(y_test, predicts)))
    print('my Precision score: ', format(precision_score(y_test, predicts)))
    print('my Recall score: ', format(recall_score(y_test, predicts)))
    print('my F1 score: ', format(f1_score(y_test, predicts)))

    # explore how reject rate affects score
    scores = ['precision', 'recall','f1_score']
    for reject_rate in np.arange(1,15,2):
        spam_filter = naive_bayes_spam_filter(float(reject_rate))
        spam_filter.fit(x_training_data, y_train.as_matrix())
        predicts = spam_filter.predict(testing_data)
        print("reject_rate: ", reject_rate)
        #print('accuracy score: ', format(accuracy_score(y_test, predicts)))
        print('precision score: ', format(precision_score(y_test, predicts)))
        print('recall score: ', format(recall_score(y_test, predicts)))
        print('F1 score: ', format(f1_score(y_test, predicts)))
