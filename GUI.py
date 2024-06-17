from re import S
import pandas as pd

import streamlit as st
from process_NLP import NLP
from Gensim import MyGensim
from Cosine import MyCosine
from ALS import MyALS
from Surprise import MySurprise
from streamlit_option_menu import option_menu

#streamlit run gui.py
#Ghi chú


gensim = MyGensim()
cosine = MyCosine()
#als = MyALS()
mysurprise = MySurprise()


st.title("BUILDING RECOMMENDER SYSTEMS")
st.write("")
st.write("")
st.write("")
menu = ["Home", "Content-based", "Collaborative"]
#choice = st.sidebar.selectbox('Menu', menu)

with st.sidebar:
    choice = option_menu("",options=["Home", "Content-based", "Collaborative", "Read-me"], 
        icons=["home", "gear", "gear", "file"], menu_icon="cast", default_index=0)

def homepage():
    st.header("HOME PAGE")
    st.divider()
    st.write("<h3>1. Introduction</h3>", unsafe_allow_html=True)
    st.write("""<div style="font-size:1.2em">This system delves into the realm of recommender systems, 
             focusing specifically on courses of Coursera recommendation, 
             and explores the latest survey and review articles. <br/>
             In this system, we use two groups of algorithms:
             <ul>
                <li>Content-based filtering: Gensim, Cosine</li>
                <li>Collaborative filtering: Surprise (SVD)</li>
             </ul>
             </div>""", unsafe_allow_html=True)    

    st.write("<h3>2. Data</h3>", unsafe_allow_html=True)
    st.write("""<div style="font-size:1.2em">
             Analytical data is gathered from 879 Coursera courses and 223,543 course reviews
             </div>""", unsafe_allow_html=True)
    st.write("")
    
    st.write("<h3>3. Team member</h3>", unsafe_allow_html=True)
    st.write("""<div> 
             Thai Tuan Quan <br/>
             Le Thi Huong Quynh<br/>
             Huynh Chi Tai
             <div>""", unsafe_allow_html=True)





#----------------------------------------------------------------------------------
def CSS():
    st.markdown("""<style>
                    .item {border: solid 1px; height:350px; margin-bottom:10px; font-size:1em; padding:3px}
                    .title {font-weight: bold; font-size:1.3em;}
                    .review {font-weight: bold;}
                    .star {font-weight: bold;}
                    .level {font-weight: bold;}
                </style>""", unsafe_allow_html=True)


#----------------------------------------------------------------------------------
def ShowData(df):
    col1, col2 = st.columns(2)

    with col1:
       for i in range(0, df.shape[0], +2):
            course = f"""
                <div class="item">
                    <div class="title">{df.iloc[i]["CourseID"]}</div>
                    <div>Unit: <span class="unit">{df.iloc[i]["CourseID"]}</span></div>
                    <div>Review: <span class="review"></span></div>
                    <div>Star: <span class="star"></span></div>
                    <div>Level: <span class="star"></span></div>
                    <div>{df.iloc[i]["Results"][0:250]}</div>                    
                </div>
            """
            st.write(course, unsafe_allow_html=True)

    with col2:
       st.header("A dog")
       st.image("https://static.streamlit.io/examples/dog.jpg")

#----------------------------------------------------------------------------------
def ShowData(df, c):    
    
    NUMBER_OF_COLUMNS = c
    columns = st.columns(spec=NUMBER_OF_COLUMNS)    

    # Loop through columns and render in each one an equal chunk of the items
    for i, col in enumerate(columns):
        idx = [i for i in range(i, df.shape[0], NUMBER_OF_COLUMNS)]        

        for j, row in df.iloc[idx].iterrows():                       
            course = f"""
                <div class="item">
                    <div class="title">{row["CourseID"]}</div>
                    <div>Unit: <span class="unit">{row["Unit"]}</span></div>
                    <div>Review: <span class="review">{row["ReviewNumber"]}</span></div>
                    <div>Star: <span class="star">{row["AvgStar"]}</span></div>
                    <div>Level: <span class="star">{row["Level"]}</span></div>
                    <div>{row["Results"][0:250]}</div>                    
                </div>
            """
            col.write(course, unsafe_allow_html=True)               

#----------------------------------------------------------------------------------
def Contentbased():
    st.header("CONTENT-BASED FILTERING")    
    text = st.text_input("**What do you want to learn:**")    
    if (text == ""):
         st.write("""<div style="font-size:1.2em">Recommend courses for you</div>""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Gensim", "Cosine"])

    with tab1:
        ShowData(gensim.recomment(text, 6), 2)

    with tab2:        
        ShowData(cosine.recomment(text, 6), 2)
        
               
#----------------------------------------------------------------------------------
def Collaborative():
    st.header("COLLABORATIVE FILTERING")
       
    text = st.text_input("**Enter userid:**")
    
    if (text == ""):
        return    
    
    userid = 0
    try:
        userid = int(text)
        if (not mysurprise.check_userid(userid)):
            st.warning("User id does not exists")
            return

        tab1, tab2 = st.tabs(["Recommended Courses", "History"])
        
        with tab1:
            ShowData(mysurprise.recomment(userid, 6), 2)  
    
        with tab2:
            ShowData(mysurprise.history(userid, 6), 2)
    except:
        st.warning("Please enter valid user id.")
    
        #st.dataframe(mysurprise.history(text, 6))
        
#----------------------------------------------------------------------------------

CSS()
if choice == 'Home':
    homepage()
    
elif choice.lower() == 'content-based':
    Contentbased()
    

elif choice.lower() == "collaborative":
    Collaborative()
    #optUser = st.selectbox("**Please select user:**", mysurprise.user())
    

elif choice.lower() == "read-me":
    st.header("VỀ ỨNG DỤNG RECOMMENDER SYSTEMS")
    
    st.write("""
            <div style="font-size:1.2em">
                Hệ thống được chia làm 2 phần:<br/>
                <ol>
                    <li><b>Người dùng chưa có tài khoản (Content-based filtering)</b>
                        <div style="font-size:1.1em">
                            <ul>
                                <li>
                                    Khi người sử dụng không nhập gì vào ô tìm kiếm, hệ thống tự gợi ý 6 khóa học cho học viên
                                </li>
                                <li>
                                    Nếu nhập vào ô tìm kiếm, hệ thống sẽ gợi ý các khóa học liên quan tới từ gợi ý trên. 
                                    Như tên khóa học, đơn vị đào tạo, trình độ, kết quả khóa học
                                </li>
                            </ul>
                        </div>
                    </li>
                    <li><b>Người dùng đã có tài khoản (Collaborative filtering)</b>
                        <div style="font-size:1.1em">
                            Hệ thống dựa vào lịch sử đánh giá của học viên về các khóa đã học để đề xuất cho học viên các khóa học phù hợp                                                            
                        </div>
                    </li>
                    <li><b>Các thành viên trong nhóm</b>
                        <ul>
                            <li>Thái Tuấn Quân</li>
                            <li>Lê Thị Hương Quỳnh</li>
                            <li>Huỳnh Chí Tài</li>
                        </ul>
                    </li>
                </ol>
            </div> 
            """, unsafe_allow_html=True)
    


