from PyPDF2 import PdfReader
import streamlit as st 
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from dotenv import load_dotenv


def process_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", 
        chunk_size=1000, 
        chunk_overlap=200, 
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # 임베팅 처리(벡터 변환), 임베딩은 HuggingFaceEmbeddings 모델을 사용
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    documents = FAISS.from_texts(chunks, embeddings)
    return documents

def main(): # streamlit 을 이용한 웹사이트 생성
    st.title("📄PDF 요약하기")
    st.divider()
    try:
        load_dotenv()

    except ValueError as e:
        st.error(str(e))
        return
    
    pdf = st.file_uploader('PDF파일을 업로드해주세요', type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = "" # 텍스트 변수에 PDF 내용을 저장
        for page in pdf_reader.pages:
            text += page.extract_text()

        documents = process_text(text)
        query = "업로드된 PDF 파일의 내용르 약 3~5문장으로 요약해주세요." # LLM 에 PDF 파일 요약 요청

        if query:
            docs = documents.similarity_search(query)
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1) 
            chain = load_qa_chain(llm) # chain type => default 가 staff 

            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=query)
                print(cost)
            
            st.subheader("---요약 결과--")
            st.write(response)

if __name__ == "__main__":
    main()


            


        


    

