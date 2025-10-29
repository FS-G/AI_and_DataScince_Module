from google import genai
import streamlit as st
from google.genai import types
import os
 

key = os.getenv("GENIMI_KEY")
client = genai.Client(api_key= key)



sys_instruct = "You are a SARCASTIC assistant. Give responses in SARCASTIC manner."



@st.cache_resource
def load_chat():
    return client.chats.create(model="gemini-2.0-flash",
                                config=types.GenerateContentConfig(
                                max_output_tokens=500,
                                temperature=1.0,
                                system_instruction=sys_instruct
                            ))

def chatbot(query):
  chat = load_chat()
  response = chat.send_message(query)
  return response.text

st.title("Demo chatbot")
user_input = st.text_area("Write your query")
if st.button("Ask!"):
  if user_input:
    response = chatbot(user_input)
    st.write(response)