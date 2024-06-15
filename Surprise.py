import pandas as pd
import numpy as np
from surprise import Reader, Dataset, BaselineOnly


class MySurprise():

    def __init__(self):
        self._dfCourse = None 
        self._dfCourse_indexed = None
        self._df = None
        self._model = None
        
        self.load()

    def load(self):
        self._df = pd.read_csv("input_data/reviews.csv")
        self._dfCourse = pd.read_csv("input_data/courses.csv")

        self._df = self._df.dropna()
        self._df["CourseNumberId"] = self._df["CourseName"].factorize()[0]
        self._df["UserIdNumber"] = self._df["ReviewerName"].factorize()[0]
        reader = Reader()
        data = Dataset.load_from_df(self._df[['UserIdNumber', 'CourseNumberId', 'RatingStar']], reader)

        self._dfCourse_indexed = self._df[["CourseNumberId", "CourseName"]].drop_duplicates()
        
        self._model = BaselineOnly()
        trainset = data.build_full_trainset()
        self._model.fit(trainset)

    def recomment(self, userid):
        dfUser = self._df[(self._df["UserIdNumber"] == userid) & (self._df["RatingStar"] >=3)]
        dfUser = dfUser.set_index('CourseNumberId')
        
        df_score = self._df[["CourseNumberId"]]
        df_score['EstimateScore'] = df_score['CourseNumberId'].apply(lambda x: self._model.predict(userid, x).est) # est: get EstimateScore
        df_score = df_score.drop_duplicates()
        df_score = df_score.sort_values(by=['EstimateScore'], ascending=False)        

        dfResult = pd.merge(df_score, self._dfCourse_indexed, left_on="CourseNumberId", right_on="CourseNumberId", how="inner")
        dfResult = pd.merge(dfResult, self._dfCourse, left_on="CourseName", right_on="CourseName", how="inner")
        return dfResult[["CourseName", "Unit", "ReviewNumber", "AvgStar", "Level"]].head()