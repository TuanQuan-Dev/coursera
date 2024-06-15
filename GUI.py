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
    choice = option_menu("",options=["Home", "Content-based", "Collaborative"], 
        icons=["home", "gear", "gear"], menu_icon="cast", default_index=0)

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
def Contentbased():
    st.header("CONTENT-BASED FILTERING")    
    text = st.text_input("What do you want to learn:")
    
    tab1, tab2 = st.tabs(["Gensim", "Cosine"])

    with tab1:                       
        st.table(gensim.recomment(text, 5))

    with tab2:        
        st.table(cosine.recomment(text, 5))
        
               
#----------------------------------------------------------------------------------
def Collaborative():
    st.header("COLLABORATIVE FILTERING")
    
    st.write("<br/><br/>", unsafe_allow_html=True);
    text = st.text_input("Enter userid:")
        
    #st.table(als.recomment(text))
    st.table(mysurprise.recomment(text))


#----------------------------------------------------------------------------------
if choice == 'Home':
    homepage()
    
elif choice.lower() == 'content-based':
    Contentbased()
    

elif choice.lower() == "collaborative":
    Collaborative()
    pass



