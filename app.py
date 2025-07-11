#UI imports
import streamlit as st

#llm imports
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

#rag imports
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

st.title("RAG Chatbot")

prompt=st.chat_input("Enter your prompt")

#create list of previous prompts
if 'messages' not in st.session_state:
    st.session_state.messages=[]

#display each prompt in list
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

@st.cache_resource
def get_vectorstore():
    pdf_name="researchpaper.pdf"
    loaders=[PyPDFLoader(pdf_name)]
    index= VectorstoreIndexCreator(
        embedding=HuggingFaceBgeEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    ).from_loaders(loaders)
    return index.vectorstore

if prompt:
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").markdown(prompt)

    sys_prompt=ChatPromptTemplate.from_template("""You are very smart and always give precide answer of everything. Answer this question:{user_prompt}. Start answer directly and no small talk.""")

    groq=ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name="llama3-8b-8192"
    )
    try:
        vectorstore=get_vectorstore()
        if vectorstore is None:
            st.error("failed to load document")
        
        chain=RetrievalQA.from_chain_type(
            llm=groq,
            chain_type='stuff',
            retriever=vectorstore.as_retriever(search_kwargs=({"k":3})),
            return_source_documents=True 
            )
        result=chain({"query":prompt})
        
        response=result["result"]
        
        # response="i am ai"
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role":"assistant","content":response})
    except Exception as e:
        st.error("Error: [{str(e)}]")
    