

from time import sleep
import streamlit as st

username = st.text_input("Username", key="username")
password = st.text_input("Password", key="password", type="password")

if username == "test" and password == "test":
        st.session_state["logged_in"] = True
        st.success("Logged in!")
        sleep(0.5)
       