from dotenv import load_dotenv
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import streamlit as st

# Setup LLM
load_dotenv('./../.env')
base_url = "http://localhost:11434"
model = 'llama3.2'

llm = ChatOllama(base_url=base_url, model=model)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. You answer question like a {role}."),
    MessagesPlaceholder(variable_name='history'),
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser()

def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, connection = "sqlite:///chat_history.db")

history_chain = RunnableWithMessageHistory(chain, get_session_history, input_messages_key='question', history_messages_key='history')

def response_streamer(session_id, role, question):
    for response in history_chain.stream(input={"role":role,"question": question},
                                         config={"configurable": {"session_id": session_id}}):
        yield response

# Setup Streamlit UI
st.title("🦜🔗 Chat App")
user_id = st.text_input("Enter your user id", "temp_chat")
assitant_role = st.text_input("Enter assistant role", "Professor")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("Start New Conversation"):
    st.session_state.chat_history = []
    history = get_session_history(user_id)
    history.clear()

if st.button("Clear Chat Panel"):
    st.session_state.chat_history = []


for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

prompt = st.chat_input("Please enter your question here...")

if prompt:
    st.session_state.chat_history.append({'role': 'user', 'content': prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(response_streamer(user_id, assitant_role, prompt))

    st.session_state.chat_history.append({'role': 'assistant', 'content': response})
