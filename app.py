import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
import os

from dotenv import load_dotenv
load_dotenv(dotenv_path="C:/Users/admin/Desktop/Projects/Langchain/.env")

os.environ['HF_TOKEN']=os.getenv("Hugging_face_token")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("ASKMYDOC-AI \n Conversational RAG with PDF uploads and chat history")
st.write("Upload Pdf's and chat with their content")


api_key=st.text_input("Enter your OpenAIAPIKey:",type="password")

if api_key:
    llm=ChatOpenAI(openai_api_key=api_key,model_name="gpt-4o")
    
    session_id=st.text_input("Session ID",value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store={}
    uploaded_files=st.file_uploader("Choose a pdf File",type="pdf",accept_multiple_files=False)
    
    if uploaded_files:
        documents = []
        temppdf = "./temp.pdf"
        
        # uploaded_files is a single file, not a list
        with open(temppdf, "wb") as file:
            file.write(uploaded_files.read()) 
        loader = PyPDFLoader(temppdf)
        pdf_documents = loader.load()
        documents.extend(pdf_documents)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)

        # Create Chroma vector store
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # Load the PDF into LangChain documents
        loader = PyPDFLoader(temppdf)
        pdf_documents = loader.load()
        documents.extend(pdf_documents)

        contextualise_q_sys_prompt=(
            
            "Given a chat history and the latest user question"
            "which might reference context in the chat history,"
            "formulate a standalone question which can be understood"
            "without the chat history. Do NOT answer the question,"
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualise_q_prompt= ChatPromptTemplate.from_messages(
            [
                ("system",contextualise_q_sys_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
        )
        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualise_q_prompt)
        system_prompt=(
            
            "you are a question answering bot"
            "use the following pieces of retrieved context to answer"
            "If you do not know say you dont know"
            "Use 3 sentences max and keep answer concise"
            "\n\n"
            "{context}"
        )
        qa_prompt= ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
        )
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        user_input=st.text_input("yorur question")
        if user_input:
            session_history=get_session_history(session_id)
            response=conversational_rag_chain.invoke(
                {"input":user_input,"chat_history": session_history.messages},
                config={
                    "configurable":{"session_id":session_id}
                },
            )
            st.write(st.session_state.store)
            st.success("assistant: " + response['answer'])
            st.write("chat history:",session_history.messages)
else:
    st.warning("ENter Openai_api_key")
