import streamlit as st
import time

from Pipeline import send_prompt

from llama_index.core.schema import QueryBundle

def get_response(prompt, BR):
    prompt = QueryBundle(prompt)
    response = send_prompt(prompt, BR)
    for word in str(response):
        yield word
        time.sleep(0.01)

st.title("Query Engine")
BR = st.text_input("Business Request Number", placeholder="BR36922") or None

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)    

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        if BR:
            response = st.write_stream(get_response(prompt, BR))   
        else:
            response = st.write_stream(get_response(prompt, None))

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})