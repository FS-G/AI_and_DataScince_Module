from google import genai
import streamlit as st
import os

key = os.getenv("GENIMI_KEY")
client = genai.Client(api_key= key)

@st.cache_resource
def load_chat():
    chat = client.chats.create(model="gemini-2.0-flash")
    return chat


chat = load_chat()








def chatbot(query):




    
    response = chat.send_message(query)
    return response

    # print(response.text)
    # for message in chat._curated_history:
    #     print(f'role - ', message.role, end=": ")
    #     print(message.parts[0].text)


user_query = st.text_area("Enter Your Query")

if st.button("ASK!"):
    if user_query:
        response =  chatbot(user_query)
        st.write(response.text)