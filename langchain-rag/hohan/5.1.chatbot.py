import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="🦜⛓️‍뭐든지 질문하세요~")
st.title('🦜⛓️‍뭐든지 질문하세요~')

def generate_response(input_text):    
    llm = ChatOpenAI(model_name="gpt-4-0613", temperature=0)
    st.info(llm.invoke(input_text).content)

with st.form('Question'):
    text = st.text_area('질문 입력: ', 'What types of text models does OpenAI provide?') # 첫 페이지가 실행될 때 보여줄 질문
    submitted = st.form_submit_button('보내기')
    generate_response(text)


