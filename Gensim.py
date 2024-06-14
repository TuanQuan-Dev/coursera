from pickle import NONE
import pandas as pd
import numpy as np
from gensim import corpora, models, similarities

from process_NLP import NLP

class MyGensim:
    
    def __init__(self):
        self._dfCourse = NONE    

        self._tfidf = NONE
        self._index = NONE
        self._dictionary = NONE        

        self._nlp = NLP()        

        self.load()
        pass
        

    def load(self):
        self._dfCourse = pd.read_csv("input_data/Courses.csv")
        self._dfCourse =  self._dfCourse.fillna({ "Level": "All", "Unit" : "Unknown", "Results": ""})
        self._dfCourse["Results"] =  self._dfCourse.apply(lambda r: r["Results"] if r["Results"] != "" else r["CourseName"], axis=1)
        
        self._dfCourse["Content"] =  self._dfCourse['CourseName'] + " " +  self._dfCourse["Unit"] + " " +  self._dfCourse["Level"] + " " +  self._dfCourse["Results"]

        self._dfCourse["token_content"] =  self._dfCourse["Content"].apply(lambda x : self._nlp.to_token(self._nlp.clean(x)))
        
        Courses =  self._dfCourse["token_content"]
        self._dictionary = corpora.Dictionary(Courses)
        feature_cnt = len(self._dictionary.token2id)
        corpus = [self._dictionary.doc2bow(text) for text in Courses]
        
        self._tfidf = models.TfidfModel(corpus)
        self._index = similarities.SparseMatrixSimilarity(self._tfidf[corpus], num_features = feature_cnt)
       
    def recomment(self, text):
        if (text == ""):
            return pd.DataFrame()
        
        text = self._nlp.to_token(self._nlp.clean(text))
        kw_vector = self._dictionary.doc2bow(text)
        sim = self._index[self._tfidf[kw_vector]]               

        data = []
        for i in range(len(sim)):
            #if (i!=index):
            data.append((i, sim[i]))
        data = sorted(data, key=lambda x: x[1], reverse=True)
        
        indexes = []
        for i, score in data[0:10]:
            indexes.append(i)

        return self._dfCourse.iloc[indexes][["CourseID", "Unit", "ReviewNumber", "AvgStar", "Level", "Results"]]