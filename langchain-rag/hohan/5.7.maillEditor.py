import streamlit as st
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title='이메일 작성 서비스! 유후~', page_icon=':robot:')
st.header('이메일 작성기')

def getEmail():
    input_text = st.text_area(label='메일 입력', label_visibility='collapsed', placeholder='당신의 메일은...', key='input_text')
    return input_text

input_text = getEmail()

# 이메일 변환 작업을 위한 템플릿 정의
query_template = """
    메일을 작성해주세요.
    아래는 이메일입니다:
    이메일: {email}
"""

from langchain import PromptTemplate
# PrompptTemplate 인스턴스 생성
prompt = PromptTemplate(
    input_variables=['email'],
    template=query_template,
)

from langchain.chat_models import ChatOpenAI
# 언어 모델을 호출합니다. 
def loadLanguageModel():
    llm = ChatOpenAI(temperature=0.0, model_name='gpt-4')
    return llm

# 예시 이메일을 표시 
st.button('*예제를 표여주세요*', type='secondary', help='봇이 작성한 메일을 확인해보세요.')
st.markdown('### 봇이 작성한 메일은: ')

if input_text:
    llm = loadLanguageModel()
    # PromptTemplate 및 언어 모델을 사용하여 이메일 형식을 지정
    propmpt_with_email = prompt.format(email=input_text)
    formatted_email = llm.predict(propmpt_with_email)
    # 서식이 지정된 이메일 표시
    st.write(formatted_email)

