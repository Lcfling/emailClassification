import sys
from time import time
sys.path.append("C:\\Users\\HP\\Desktop\\ML Code\\")
import pickle
#import _pickle as cPickle
#using the Gaussian Bayes algorithm for classification of emails.
#the algorithm is imported from the sklearn library
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from email_preprocess import preprocess
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import codecs
import tokenize


if __name__ == '__main__':
    features_train, features_test, labels_train, labels_test = preprocess('./test/content.pkl', './test/index.pkl')
    # 算法选取 当前选取贝叶斯
    clf = GaussianNB()

    t0 = time()
    clf.fit(features_train, labels_train)
    with open("models/Bayes.pkl", "wb") as f:
        pickle.dump(clf, f)

    with open("models/Bayes.pkl", "rb") as t:
        clf2 = pickle.load(t)

    print("\nTraining time:", round(time() - t0, 3), "s\n")
    t1 = time()
    pred = clf2.predict(features_test)
    print("Predicting time:", round(time() - t1, 3), "s\n")
    #对验证集进行测评 获取识别成功率
    print("Accuracy of Naive Bayes: ", accuracy_score(pred, labels_test))