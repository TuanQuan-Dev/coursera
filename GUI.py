import pandas as pd

import streamlit as st
from process_NLP import NLP
from Gensim import MyGensim
from Cosine import MyCosine
from streamlit_option_menu import option_menu

#streamlit run gui.py
# Ghi chú

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
               
        gensim = MyGensim()
        st.table(gensim.recomment(text, 5))

    with tab2:
        #cosine = MyCosine()
        #st.table(cosine.recomment(text, 5))
        pass 
               
    


#----------------------------------------------------------------------------------
if choice == 'Home':
    homepage()
    
elif choice.lower() == 'content-based':
    Contentbased()
    

elif choice.lower() == "collaborative":
    st.subheader("1. Text")
    name = st.text_input("Enter your name")
    st.write(f"Your name is {name}")
    
    data = {'Name': ['Tom', 'nick', 'krish', 'jack'],
        'Age': [20, 21, 19, 18]}
 
    # Create DataFrame
    df = pd.DataFrame(data)
    for idx, row in df.iterrows():
        st.write(row["Name"])



