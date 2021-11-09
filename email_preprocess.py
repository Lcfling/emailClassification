#!/usr/bin/python

import _pickle as cPickle
import numpy

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif



def preprocess(words_file = "D:\\email\\emailClassification\\word_data_new.pkl", authors_file="D:\\email\\emailClassification\\email_authors.pkl"):
    """ 
        this function takes a pre-made list of email texts (by default word_data.pkl)
        and the corresponding authors (by default email_authors.pkl) and performs
        a number of preprocessing steps:
            -- splits into training/testing sets (10% testing)
            -- vectorizes into tfidf matrix
            -- selects/keeps most helpful features

        after this, the feaures and labels are put into numpy arrays, which play nice with sklearn functions

        4 objects are returned:
            -- training/testing features
            -- training/testing labels

    """

    ### the words (features) and authors (labels), already largely preprocessed
    ### this preprocessing will be repeated in the text learning mini-project
    authors_file_handler = open(authors_file, "rb")
    authors = cPickle.load(authors_file_handler)

    authors_file_handler.close()

    words_file_handler = open(words_file, "rb")
    word_data = cPickle.load(words_file_handler)

    words_file_handler.close()

    ### test_size is the percentage of events assigned to the test set
    ### (remainder go into training)



    features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.2, random_state=42)


    ### text vectorization--go from strings to lists of numbers
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    vmodel=vectorizer.fit(features_train)


    #保存TF-idf模型
    with open("models/vmodel.pickle", "wb") as f:
        cPickle.dump(vmodel, f)

    with open("models/vmodel.pickle", "rb") as t:
        vmodel2 = cPickle.load(t)



    features_train_transformed= vmodel2.transform(features_train)
    features_test_transformed  = vmodel2.transform(features_test)

    ### feature selection, because text is super high dimensional and 
    ### can be really computationally chewy as a result
    selector = SelectPercentile(f_classif, percentile=10)
    semodel= selector.fit(features_train_transformed, labels_train)

    # 保存特征值选取模型
    with open("models/semodel.pickle", "wb") as f:
        cPickle.dump(semodel, f)

    with open("models/semodel.pickle", "rb") as t:
        semodel2 = cPickle.load(t)
    features_train_transformed = semodel2.transform(features_train_transformed).toarray()
    features_test_transformed  = semodel2.transform(features_test_transformed).toarray()

    ### info on the data
    print("\nno. of Chris training emails:", sum(labels_train))
    print("\nno. of Sara training emails:", len(labels_train)-sum(labels_train))


    
    return features_train_transformed, features_test_transformed, labels_train, labels_test
