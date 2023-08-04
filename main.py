import os 
from pandasai import PandasAI
from pandasai.llm.starcoder import Starcoder
from dotenv import load_dotenv
import streamlit as st
import pandas as pd

def main():
    load_dotenv()
    API_KEY = os.environ["HUGGINGFACEHUB_API_KEY"]
    llm = Starcoder(api_token=API_KEY)
    pandas_ai = PandasAI(llm=llm)
    st.title("Prompt-driven Analysis with PandasAI 📊")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file for analysis", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df.head(3))
        prompt = st.text_area("Enter your prompt : ")
        if st.button("Generate"):
            if prompt:
                response = pandas_ai.run(df, prompt=prompt)
                st.write(response)
            else:
                st.write("Please enter a prompt first.")
    return

if __name__ == '__main__':
    main()