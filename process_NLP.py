import numpy as np
import pandas as pd
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

class NLP:
    
    def __init__(self):
        self.emoji_dict = {}
        self.custom_stopwords_lst = []
        
        self.load()
        pass

#---------------------------------------------------------------------------------------------------
# lấy dữ liệu từ file, cho việc chuẩn hóa    
    def load(self):
        ##lấy EMOJICON
        file = open("input_data/files/emojicon_replace.txt", "r", encoding="utf8")
        emoji_lst = file.read().split("\n")        
        
        for line in emoji_lst:
            key, value = line.split("\t")
            self.emoji_dict[key] = str(value)
        file.close()

        ##lấy EMOJICON
        file = open("input_data/files/custom_stopwords.txt", "r", encoding="utf8")
        self.custom_stopwords_lst = file.read().split("\n")
        file.close()
        
#---------------------------------------------------------------------------------------------------
# hàm dùng để chuẩn hóa phần text của khóa học
    def clean(self, text):
    # chuyển thành kí tự thường
        text = text.lower()

    # Bỏ các ký tự đặc biệt
        PUNCTUATION = "!\"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~`0123456789"
        #text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.translate(str.maketrans('', '', PUNCTUATION))
        
    # Bỏ stopword
        STOPWORDS = set(stopwords.words('english'))
        text = " ".join([word for word in str(text).split() if word not in STOPWORDS])

    # Bỏ các từ emojion        
        file = open("input_data/files/emojicon.txt", "r", encoding="utf8")
        emoji_lst = file.read().split("\n")
        text = " ".join([word for word in str(text).split() if word not in emoji_lst])
        return text

#---------------------------------------------------------------------------------------------------
# hàm này dành để xử lý text cho phần bình luận
    def clean_full(self, text):
    # chuyển thành kí tự thường
        text = text.lower()

    # Bỏ các ký tự đặc biệt
        PUNCTUATION = "!\"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~`0123456789"
        #text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.translate(str.maketrans('', '', PUNCTUATION))
        
    # Bỏ stopword
        STOPWORDS = set(stopwords.words('english'))
        text = " ".join([word for word in str(text).split() if word not in STOPWORDS])

    # đổi các emojion sang từ tương đương
        text = ''.join(self.emoji_dict[word] + " " if word in self.emoji_dict else word for word in list(text))

     # thay các từ trong danh sách stopword
        text = " ".join([word for word in str(text).split() if word not in self.custom_stopwords_lst])
        
        return text

#---------------------------------------------------------------------------------------------------
# chuyển text thành các từ riêng lẻ
    def to_token(self, text):
        return word_tokenize(text)
