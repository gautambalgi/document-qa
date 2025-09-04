import streamlit as st
import lab1
import lab2

st.set_page_config(page_title="Multi-Lab App", layout="wide")

nav = st.navigation([
    st.Page("lab2.py", title="Lab 2", ),  # now default
    st.Page("lab1.py", title="Lab 1", ),
])

nav.run()


