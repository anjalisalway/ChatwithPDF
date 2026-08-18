import os
import tempfile
import streamlit as st

from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

st.set_page_config(page_title="RAG Chatbot - Research Paper Q&A")
st.title("RAG Chatbot — Research Paper Q&A")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'pdf_path' not in st.session_state:
    st.session_state['pdf_path'] = None

if 'vectorstore' not in st.session_state:
    st.session_state['vectorstore'] = None


def create_vectorstore_from_pdf(pdf_path):
    """Load PDF, split text, create vectorstore index and return the vectorstore."""
    loaders = [PyPDFLoader(pdf_path)]
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    ).from_loaders(loaders)
    return index.vectorstore


# Upload area
st.sidebar.header("Upload")
uploader = st.sidebar.file_uploader("Upload research paper (PDF)", type=["pdf"])

if uploader is not None:
    # Save uploaded file to a temp file and remember path in session state
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploader.read())
        tmp_path = tmp.name
    st.session_state['pdf_path'] = tmp_path
    st.sidebar.success("PDF uploaded and saved.")

# If a PDF path exists and no vectorstore built yet (or file changed), build it
if st.session_state.get('pdf_path') and (st.session_state.get('vectorstore') is None):
    try:
        with st.spinner('Processing PDF and building vector store...'):
            vs = create_vectorstore_from_pdf(st.session_state['pdf_path'])
            st.session_state['vectorstore'] = vs
        st.success('Document processed and vector store created.')
    except Exception as e:
        st.error(f"Failed to create vectorstore from uploaded PDF: {e}")

# Show uploaded file info
if st.session_state.get('pdf_path'):
    st.sidebar.markdown(f"**Current PDF:** {st.session_state['pdf_path']}")
else:
    st.sidebar.info('No PDF uploaded yet. Upload a research paper to enable RAG queries.')

# Display previous chat messages
for message in st.session_state['messages']:
    st.chat_message(message['role']).markdown(message['content'])

# Input area (use chat_input if available, else text_input)
prompt = st.chat_input("Enter your question about the uploaded paper")

if prompt:
    st.session_state['messages'].append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # Ensure vectorstore is available
    if st.session_state.get('vectorstore') is None:
        st.error('No document has been processed yet. Please upload a PDF and wait for processing.')
    else:
        try:
            groq = ChatGroq(
                groq_api_key=os.environ.get('GROQ_API_KEY'),
                model_name=os.environ.get('GROQ_MODEL', 'llama3-8b-8192')
            )

            chain = RetrievalQA.from_chain_type(
                llm=groq,
                chain_type='stuff',
                retriever=st.session_state['vectorstore'].as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )

            with st.spinner('Running retrieval + LLM...'):
                result = chain({"query": prompt})

            # result usually contains 'result' and 'source_documents'
            response_text = result.get('result') or result.get('answer') or str(result)

            st.chat_message("assistant").markdown(response_text)
            st.session_state['messages'].append({"role": "assistant", "content": response_text})

            # Optionally show source docs
            if result.get('source_documents'):
                st.markdown('---')
                st.markdown('**Source documents / retrieved chunks:**')
                for i, doc in enumerate(result['source_documents']):
                    meta = getattr(doc, 'metadata', {})
                    page_info = meta.get('page', '') or meta.get('source', '')
                    st.write(f"[{i+1}] {page_info}")
                    st.write(doc.page_content[:1000])

        except Exception as e:
            st.error(f"Error during pipeline execution: {e}")
    