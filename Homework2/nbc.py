import os
import nltk
import math
import pandas as pd
import numpy as np
import string
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
from sklearn.metrics import accuracy_score

data = 'D:/vsm+knn/data/20news-18828'
data_train = 'D:/vsm+knn/data/data_train'
data_test = 'D:/vsm+knn/data/data_test'
_dictionary = 'D:/vsm+knn/data/dictionary.csv'


def pre(input):
    # 读取数据
    raw_data = []
    sort = []
    num = 0
    for file1 in os.listdir(input):
        path1 = os.path.join(input, file1)
        num += 1
        for file2 in os.listdir(path1):
            path2 = os.path.join(path1, file2)
            sort.append(num)
            with open(path2, encoding='latin-1') as file:
                document = file.read()
                raw_data.append(document)
    # 处理数据
    new_data = []
    for doc in raw_data:
        delpunctuation = re.compile('[%s]' % re.escape(string.punctuation))
        doc = delpunctuation.sub("", doc)
        lowers = str(doc).lower()
        tokens = nltk.word_tokenize(lowers)
        stemmer = PorterStemmer()
        stoplist = stopwords.words('english')
        words = []
        for word in tokens:
            if word not in stoplist:
                words.append(stemmer.stem(word))
        new_data.append(words)
    return new_data, sort


def NBC(train_data, train_label, test_data, test_label):
    dictionary = np.array(pd.read_csv(_dictionary, sep=" ", header=None)).reshape(1, -1)[0]
    kindcount = []
    allcount = []
    kindnum = []
    kindcounte = []
    Accuracy = []
    for i in range(len(train_data)):
        doc = list(filter(lambda word: word in dictionary, train_data[i]))
        if train_label[i] < len(kindcount):
            kindcount[int(train_label[i])] += doc
            kindnum[int(train_label[i])] += 1
        else:
            kindcount.append(doc)
            kindnum.append(1)
        allcount += doc
    allcounter = Counter(allcount)
    for kind in range(20):
        kindcounte.append(Counter(kindcount[kind]))
    for i in range(len(test_data)):
        Acc = []
        for kind in range(20):
            Pxy = 0
            Px = 0
            kindcounter = kindcounte[kind]
            for word in test_data[i]:
                if word in dictionary:
                    P0 = math.log((kindcounter[word] + 1) / (len(kindcount[kind]) + len(dictionary)))
                    Pxy += P0
                    P1 = math.log((allcounter[word] + 1) / (len(allcount[kind]) + len(dictionary)))
                    Px += P1
            Py = math.log(kindnum[kind] / 18828.0)
            Pyx = Pxy + Py - Px
            Acc.append([kind, Pyx])
        Acc = sorted(Acc, key=lambda item: -item[1])
        Accuracy.append(Acc[0][0])
    print(" Accuracy:\t", accuracy_score(test_label, Accuracy))


if __name__ == '__main__':
    train_data, train_label = pre(data_train)
    test_data, test_label = pre(data_test)
    NBC(train_data, train_label, test_data, test_label)
