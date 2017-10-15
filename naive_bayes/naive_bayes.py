#!/usr/bin/env python

import numpy as np
import pandas as pd


class naive_bayes_spam_filter:

    def __init__(self,training_data,labels):

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
        self.reject = 10  # initial reject rate  p(spam|x)/p(ham|x)

    def fit(self, training_data, labels):
        """
        return matrix of spam/ham for the document
        :param training_data: bags of words   [num_doc, num_words]  training_data[i,j] = frequency of jth word in ith document
        :param labels:  labels [num_doc]  labels[i] = 1/0 spam /ham
        :return self
        """
        # Count ham/spam occurrence of each word
        # optimize it! very slow
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

        # probability of document being classified as spam P = p(spam|w1) * p(spam|w2) * ...*p(spam|wn)
        # log(P) = sigma(log (p(wi|spam) / (p(wi)/p(spam))
        # but we don't know the p(wi) so approximately p(wi)/p(spam) ~= p(wi|ham)

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

    df = pd.read_table('smsspamcollection/SMSSpamCollection',
                       sep='\t',
                       header=None,
                       names=['label', 'sms_message'])
    df['label'] = df.label.map({'ham': 0, 'spam': 1})

    # split to training and testing set
    X_train, X_test, y_train, y_test = train_test_split(df['sms_message'],
                                                        df['label'],
                                                        random_state=1)

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
    spam_filter = naive_bayes_spam_filter(x_training_data, y_train.as_matrix())
    spam_filter.fit(x_training_data,y_train.as_matrix())

    predicts = spam_filter.predict(testing_data)
    print("my prediction: ", predicts)
    print('my Accuracy score: ', format(accuracy_score(y_test, predicts)))
    print('my Precision score: ', precision_score(y_test, predicts))
    print('my Recall score: ', format(recall_score(y_test, predicts)))
    print('my F1 score: ', format(f1_score(y_test, predicts)))


