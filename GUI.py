import pandas as pd

import streamlit as st
from process_NLP import NLP
from Gensim import MyGensim


#streamlit run gui.py
# Ghi chú

st.title("Trung Tâm Tin Hoc")
st.subheader("How to run streamlit app 1")
menu = ["Home", "About", "Content"]
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Home':
    st.subheader("Streamlit From Windows")
elif choice == 'About':
    st.subheader("[Trung Tam Tin Hoc](https://csc.edu.vn)")
    
    nlp = NLP()
    text = st.text_input("Enter your text")
    st.write(f"Text alter clean {nlp.clean(text)}")
    
    gensim = MyGensim()
    st.table(gensim.recomment(text))

elif choice == "Content":
    st.subheader("1. Text")
    name = st.text_input("Enter your name")
    st.write(f"Your name is {name}")
    
    data = {'Name': ['Tom', 'nick', 'krish', 'jack'],
        'Age': [20, 21, 19, 18]}
 
    # Create DataFrame
    df = pd.DataFrame(data)
    
    st.table(df)
