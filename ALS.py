import findspark
findspark.init()

import pyspark
from pyspark import SparkContext
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer

from pyspark.ml.recommendation import ALS


class MyALS():
    def __init__(self):

        self._user_recs = None
        self._dfCourse = None

        self.load()
        pass

    def load(self):
        spark = SparkSession.builder.appName("GUI").getOrCreate()

        self._dfCourse = spark.read.csv("input_data/courses.csv", header=True, inferSchema=True)
        df = spark.read.csv("input_data/reviews.csv", header=True, inferSchema=True)
        df = df.withColumn("ReviewerName", when(col("ReviewerName").startswith("By"), col("ReviewerName"))
                             .otherwise("Others"))
        df = df.withColumn("RatingStar", df["RatingStar"].astype("Integer"))
        df = df.drop_duplicates()
        df = df.withColumn("DateOfReview", to_date("DateOfReview", "MMM d, yyyy"))
        df = df.dropna(how="any", subset= ['CourseName', 'ReviewerName'])
        df = df.withColumn("RatingStar",
                         when(col("RatingStar").isNull(), 0)
                         .when(~(col("RatingStar").cast("double").isNotNull()), 1)
                         .otherwise(col("RatingStar")))
        df = df.withColumn("ReviewContent",
                         when(col("ReviewContent").isNull(), "None")
                         .when((col("ReviewContent").cast("double").isNotNull()), "None")
                         .otherwise(col("ReviewContent")))

        max_date = df.select(max('DateOfReview')).collect()[0][0]

        # Điền ngày tháng gần nhất cho mỗi row
        df = df.withColumn("DateOfReview", when(col("DateOfReview").isNull(), max_date)
                                 .otherwise(col("DateOfReview")))

        # Định dạng lại cột date
        df = df.withColumn("DateOfReview", date_format("DateOfReview", "yyyy-MM-dd"))
        df = df.filter((df.RatingStar >0 ) & (df['ReviewerName'] != "Others")).groupBy(['CourseName','ReviewerName']).agg(avg('RatingStar').alias('AvgRating'))

        indexer_coursename = StringIndexer(inputCol='CourseName', outputCol='CourseName_idx')
        data_indexed = indexer_coursename.fit(df).transform(df)

        indexer_reviewer = StringIndexer(inputCol='ReviewerName', outputCol='ReviewerName_idx')        
        data_indexed =  indexer_reviewer.fit(data_indexed).transform(data_indexed)

        self._data_course = data_indexed.select(["CourseName", "CourseName_idx"]).drop_duplicates()
        self._data_course = self._data_course.withColumnRenamed("CourseName", "CourseName_ref")
        
        train_data, test_data = data_indexed.randomSplit([0.8, 0.2])
        als = ALS(maxIter=10, regParam=0.09, rank = 25, userCol="ReviewerName_idx", itemCol="CourseName_idx",
                      ratingCol="AvgRating", coldStartStrategy="drop", nonnegative=True)
        model = als.fit(train_data)

        # mỗi user lấy 5 khóa
        self._user_recs = model.recommendForAllUsers(5)

    def recomment(self, id):
        result = self._user_recs.filter(self._user_recs['ReviewerName_idx'] == id)
        result = result.select(result.ReviewerName_idx, explode(result.recommendations))
        result = result.withColumn('CourseName_idx', result.col.getField('CourseName_idx'))\
                                .withColumn('rating',result.col.getField('rating'))

        result = result.join(self._data_course, on="CourseName_idx", how ="inner")
        result = result.join(self._dfCourse, result.CourseName_ref == self._dfCourse.CourseName, how ="inner")
        
        return result.select(["CourseName", "Unit", "ReviewNumber", "AvgStar", "Level"]).toPandas()
    
