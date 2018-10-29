import os
import nltk
import pandas as pd
import string
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter

input = 'D:/vsm+knn/data/20news-18828'
output = 'D:/vsm+knn/data/vsm.csv'

def vsm():
    # 读取数据
    raw_data = []
    for file1 in os.listdir(input):
        path1 = os.path.join(input, file1)
        for file2 in os.listdir(path1):
            path2 = os.path.join(path1, file2)
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
    # 创建字典
    dictionary = []
    count = []
    for word in new_data:
        count += word
    count = Counter(count)
    for word in count:
        if count[word] >= 9 and count[word] <= 10000:
            if word not in dictionary:
                dictionary.append(str(word))
    # 生成VSM
    vectors = []
    for words in new_data:
        vector = []
        for word in dictionary:
            if word in words:
                vector.append('1')
            else:
                vector.append('0')
        vectors.append(vector)
    pd.DataFrame(vectors).to_csv(output, sep=",", header=None, index=None)

if __name__ == '__main__':
    vsm()