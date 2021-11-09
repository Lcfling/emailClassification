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
    #加载识别模型
    with open("models/Bayes.pkl", "rb") as t:
        clf2 = pickle.load(t)

    words_file = "./test/content.pkl"
    words_file_handler = open(words_file, "rb")
    word_data = pickle.load(words_file_handler)

    words_file_handler.close()

    ## testData 待分类预测的数据
    testData = [word_data[44], word_data[1], word_data[2], word_data[3]]

    #加载TF-idf词频特征模型
    with open("models/vmodel.pickle", "rb") as t:
        vmodel2 = pickle.load(t)
    features_train_transformed = vmodel2.transform(testData)

    #加载特征提取模型
    with open("models/semodel.pickle", "rb") as t:
        semodel2 = pickle.load(t)
    features_train_transformed = semodel2.transform(features_train_transformed).toarray()

    res = clf2.predict(features_train_transformed)

    print("单项测试集result")
    print(res)