import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

from process_NLP import NLP
    

# định nghĩa 1 lớp MyCosin
class MyCosine:
    
    def __init__(self):
        self._dfCourse = None    

        self._vectorizer = None
        self._vectorizer_matrix = None        

        self._nlp = NLP()        

        self.load()
        pass
    

# đọc dữ liệu, chuẩn bị dữ liệu
    def load(self):
        self._dfCourse = pd.read_csv("input_data/courses.csv")
        self._dfCourse =  self._dfCourse.fillna({ "Level": "All", "Unit" : "Unknown", "Results": ""})
        self._dfCourse["Results"] =  self._dfCourse.apply(lambda r: r["Results"] if r["Results"] != "" else r["CourseName"], axis=1)
        
        self._dfCourse["Content"] =  self._dfCourse['CourseName'] + " " +  self._dfCourse["Unit"] + " " +  self._dfCourse["Level"] + " " +  self._dfCourse["Results"]

        self._dfCourse["clean_content"] =  self._dfCourse["Content"].apply(lambda x : self._nlp.clean(x))
        
        # vectorizer dữ liệu
        self._vectorizer = TfidfVectorizer(analyzer="word", min_df=10)
        self._vectorizer_matrix =  self._vectorizer.fit_transform(self._dfCourse["clean_content"])
        
        # Lưu vectorizer
        # with open("output_data/vectorizer.pkl", "wb") as file:
        #     pickle.dump(vectorizer, file)
    

# gợi ý khóa học theo text nhập
    def recomment(self, text, n):
        if (text == ""):
            return pd.DataFrame()
        
        text = self._nlp.clean(text)
        vectorizer_matrix_test = self._vectorizer.transform([text])
        
        # tạo ma trận liên quan
        cosine_similarities = cosine_similarity(vectorizer_matrix_test, self._vectorizer_matrix)
        
        i = 0
        results = []
        for x in cosine_similarities[0]:
            results.append((i, x))
            i += 1
        
        # sắp xếp kết quả
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        indexes = []
        for idx, value in sorted_results[:n]:
            indexes.append(idx)
        return self._dfCourse.iloc[indexes][["CourseID", "Unit", "ReviewNumber", "AvgStar", "Level", "Results"]].reset_index(drop=True)