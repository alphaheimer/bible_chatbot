import os 
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
import pinecone
from langchain import OpenAI

pinecone.init("4f028635-3471-4227-9e8b-d3b876dbfa37",environment="asia-southeast1-gcp-free") 

def run_llm(query):
    embeddings = OpenAIEmbeddings(openai_api_key='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')
    docsearch = Pinecone.from_existing_index(index_name='bible-chat',embedding=embeddings) 
    chat= ChatOpenAI(openai_api_key='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',verbose=True,temperature=0)
    qa = RetrievalQA.from_llm(llm=chat,retriever=docsearch.as_retriever())
    return qa({'query':query})   


import streamlit as st

from streamlit_chat import message
st.header("BIBLE CHAT") 
prompt = st.text_input("PROMPT",placeholder="Ask me anything about the Bible")


if prompt:
    with st.spinner("Generating Response ....."):
        generated_repsonse = run_llm(query=prompt)
        print(generated_repsonse) 
        message(prompt,is_user=True) 
        message(generated_repsonse['result']) 
# right now we have not implemented memeory in our application therefore we need to have memory added to our data 






