import pandas as pd
import numpy as np
from surprise import SVD, Reader, Dataset, BaselineOnly


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
        self._dfCourse["Results"] = self._dfCourse["Results"].astype(str)

        self._df = self._df.dropna()
        self._df["CourseNumberId"] = self._df["CourseName"].factorize()[0]
        self._df["UserIdNumber"] = self._df["ReviewerName"].factorize()[0]
        reader = Reader()
        data = Dataset.load_from_df(self._df[['UserIdNumber', 'CourseNumberId', 'RatingStar']], reader)

        self._dfCourse_indexed = self._df[["CourseNumberId", "CourseName"]].drop_duplicates()
        
        self._model = SVD()
        trainset = data.build_full_trainset()
        self._model.fit(trainset)

# get list user
    def user(self):
        dfResult = self._df["ReviewerName"].drop_duplicates()
        return dfResult


# check userid
    def check_userid(self, userid):
        dfResult = self._df[self._df["UserIdNumber"] == int(userid)]
        if (dfResult.shape[0] > 0):
            return True
        return False

# get history course of user
    def history(self, userid, n):
              
        dfResult = self._df[self._df["UserIdNumber"] == int(userid)]
        dfResult = pd.merge(dfResult, self._dfCourse, left_on="CourseName", right_on="CourseName", how="inner")
        dfResult = dfResult.sort_values(by=["DateOfReview"], ascending=False)  
        return dfResult[["CourseID", "Unit", "ReviewNumber", "AvgStar", "Level", "Results"]].head(n)        


# recoment new courses
    def recomment(self, userid, n):
        dfUser = self._df[(self._df["UserIdNumber"] == userid) & (self._df["RatingStar"] >=3)]
        dfUser = dfUser.set_index("CourseNumberId")
        
        df_score = self._df[["CourseNumberId"]]
        df_score['EstimateScore'] = df_score['CourseNumberId'].apply(lambda x: self._model.predict(userid, x).est) # est: get EstimateScore
        df_score = df_score.sort_values(by=['EstimateScore'], ascending=False)  
        df_score = df_score.drop_duplicates()              

        dfResult = pd.merge(df_score, self._dfCourse_indexed, left_on="CourseNumberId", right_on="CourseNumberId", how="inner")
        dfResult = pd.merge(dfResult, self._dfCourse, left_on="CourseName", right_on="CourseName", how="inner")
        return dfResult[["CourseID", "Unit", "ReviewNumber", "AvgStar", "Level", "Results"]].head(n)